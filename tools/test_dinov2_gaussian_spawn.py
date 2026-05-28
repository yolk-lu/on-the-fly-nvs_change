from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from pipeline.place_recognition import DINOv2GlobalDescriptorExtractor


def _load_image(path: Path, max_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if max_size > 0:
        scale = min(1.0, float(max_size) / float(max(image.size)))
        if scale < 1.0:
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _save_heatmap(score: torch.Tensor, path: Path) -> None:
    score = score.detach().cpu().float()
    score = (score - score.min()) / (score.max() - score.min()).clamp_min(1e-8)
    arr = (score.numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_overlay(image: torch.Tensor, uv: torch.Tensor, path: Path, color: tuple[int, int, int]) -> None:
    arr = (image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    uv_cpu = uv.detach().cpu().float()
    for x, y in uv_cpu.tolist():
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=color)
    pil.save(path)


def _normalize(score: torch.Tensor) -> torch.Tensor:
    score = score.float()
    return ((score - score.min()) / (score.max() - score.min()).clamp_min(1e-8)).clamp(0, 1)


def _laplacian_score(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=0, keepdim=True)[None]
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    lap = F.conv2d(gray, kernel, padding=1).abs()[0, 0]
    return _normalize(F.avg_pool2d(lap[None, None], 5, stride=1, padding=2)[0, 0])


def _dino_score(extractor: DINOv2GlobalDescriptorExtractor, image: torch.Tensor, allow_fallback: bool) -> torch.Tensor:
    if extractor.model is None and not allow_fallback:
        raise RuntimeError("DINOv2 model was not loaded. Use --allow_fallback only for CPU contract checks.")
    tokens = extractor.extract_tokens(image).detach().float().cpu()
    n_tokens = int(tokens.shape[0])
    side = int(round(math.sqrt(n_tokens)))
    if side * side == n_tokens:
        token_map = tokens.reshape(side, side, tokens.shape[-1]).permute(2, 0, 1)[None]
    else:
        token_map = tokens.reshape(1, n_tokens, tokens.shape[-1]).permute(2, 0, 1)[None]
    centered = token_map - token_map.mean(dim=(2, 3), keepdim=True)
    score = centered.square().mean(dim=1, keepdim=True).sqrt()
    score = F.interpolate(score, image.shape[-2:], mode="bilinear", align_corners=True)[0, 0]
    return _normalize(score)


def _sample_uv(score: torch.Tensor, target: int) -> torch.Tensor:
    h, w = score.shape
    flat = score.flatten()
    eligible = torch.isfinite(flat) & (flat > 0)
    if int(eligible.sum().item()) == 0:
        return torch.empty(0, 2)
    k = min(int(target), int(eligible.sum().item()))
    top = torch.topk(flat.masked_fill(~eligible, -1.0), k=k, largest=True).indices
    y = torch.div(top, w, rounding_mode="floor")
    x = top - y * w
    return torch.stack([x, y], dim=-1).float()


def _coverage_stats(uv: torch.Tensor, width: int, height: int, grid: int = 16) -> dict:
    if uv.numel() == 0:
        return {"image_coverage_ratio": 0.0, "grid_coverage_ratio": 0.0}
    cells_x = torch.clamp((uv[:, 0] / max(width, 1) * grid).long(), 0, grid - 1)
    cells_y = torch.clamp((uv[:, 1] / max(height, 1) * grid).long(), 0, grid - 1)
    occupied = torch.unique(cells_y * grid + cells_x)
    bbox_area = ((uv[:, 0].max() - uv[:, 0].min() + 1) * (uv[:, 1].max() - uv[:, 1].min() + 1)).item()
    return {
        "image_coverage_ratio": float(bbox_area / max(width * height, 1)),
        "grid_coverage_ratio": float(occupied.numel() / float(grid * grid)),
    }


def _write_ply(path: Path, uv: torch.Tensor, image: torch.Tensor, focal: float, depth: float) -> dict:
    if uv.numel() == 0:
        return {"xyz_cam_bbox": []}
    h, w = image.shape[-2:]
    centre = torch.tensor([0.5 * (w - 1), 0.5 * (h - 1)], dtype=torch.float32)
    z = torch.full((uv.shape[0], 1), float(depth))
    xy = (uv - centre[None]) / max(float(focal), 1e-6) * z
    xyz = torch.cat([xy, z], dim=1)
    colors = image[:, uv[:, 1].long().clamp(0, h - 1), uv[:, 0].long().clamp(0, w - 1)].T
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz.tolist(), (colors * 255).byte().tolist()):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
    return {
        "xyz_cam_bbox": {
            "min": xyz.min(dim=0).values.tolist(),
            "max": xyz.max(dim=0).values.tolist(),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DINOv2/Laplacian Gaussian spawn candidates.")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-i", "--images_dir", default="images")
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--frames", default="0", help="Comma-separated image indices, e.g. 0,50,100")
    parser.add_argument("--max_size", type=int, default=960)
    parser.add_argument("--target", type=int, default=8192)
    parser.add_argument("--descriptor_dim", type=int, default=64)
    parser.add_argument("--allow_fallback", action="store_true")
    parser.add_argument("--constant_depth", type=float, default=1.0)
    args = parser.parse_args()

    image_dir = Path(args.source_path) / args.images_dir
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    frame_ids = [int(item) for item in args.frames.split(",") if item.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DINOv2GlobalDescriptorExtractor(descriptor_dim=args.descriptor_dim, use_dinov2=True, device=device)
    summary = []
    for frame_id in frame_ids:
        image_path = images[frame_id]
        image = _load_image(image_path, args.max_size)
        dino = _dino_score(extractor, image.to(device), allow_fallback=args.allow_fallback).cpu()
        lap = _laplacian_score(image)
        combined = _normalize(0.6 * dino + 0.4 * lap)
        modes = {"dino": dino, "laplacian": lap, "combined": combined}
        Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(out_dir / f"frame_{frame_id:04d}_rgb.png")
        frame_stats = {"frame_id": frame_id, "image": str(image_path), "height": image.shape[-2], "width": image.shape[-1]}
        for name, score in modes.items():
            uv = _sample_uv(score, args.target)
            _save_heatmap(score, out_dir / f"frame_{frame_id:04d}_{name}_score.png")
            _save_overlay(image, uv, out_dir / f"frame_{frame_id:04d}_{name}_spawn_overlay.png", (255, 32, 32))
            stats = {
                "spawned": int(uv.shape[0]),
                "score_mean": float(score.mean().item()),
                "score_max": float(score.max().item()),
                **_coverage_stats(uv, image.shape[-1], image.shape[-2]),
                **_write_ply(out_dir / f"frame_{frame_id:04d}_{name}_spawn.ply", uv, image, max(image.shape[-2:]), args.constant_depth),
            }
            frame_stats[name] = stats
        with open(out_dir / f"frame_{frame_id:04d}_spawn_stats.json", "w") as f:
            json.dump(frame_stats, f, indent=2)
        summary.append(frame_stats)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
