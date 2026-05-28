#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch


def list_images(image_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    names = [n for n in os.listdir(image_dir) if os.path.splitext(n)[1].lower() in exts]
    names.sort()
    return [os.path.join(image_dir, n) for n in names]


def c2w_from_w2c(extri_3x4: torch.Tensor) -> torch.Tensor:
    R_cw = extri_3x4[:, :3]
    t_cw = extri_3x4[:, 3]
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    T = torch.eye(4, dtype=extri_3x4.dtype, device=extri_3x4.device)
    T[:3, :3] = R_wc
    T[:3, 3] = t_wc
    return T


def w2c_from_c2w(T_wc: np.ndarray) -> np.ndarray:
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw
    return T_cw


def quat_xyzw_from_rot(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export VGGT camera-only trajectory prior (chunked) to TUM/CSV."
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Folder containing ordered image sequence.")
    parser.add_argument("--output_tum", type=str, required=True, help="Output TUM trajectory path.")
    parser.add_argument("--output_csv", type=str, default="", help="Optional CSV output path.")
    parser.add_argument(
        "--output_camera_csv",
        type=str,
        default="",
        help="Optional camera-parameter CSV (W2C extrinsics + intrinsics). Defaults next to output_tum.",
    )
    parser.add_argument(
        "--output_camera_npz",
        type=str,
        default="",
        help="Optional camera-parameter NPZ for matrix-level downstream use.",
    )
    parser.add_argument("--chunk_size", type=int, default=16, help="Images per VGGT forward chunk.")
    parser.add_argument("--overlap", type=int, default=4, help="Chunk overlap for trajectory stitching.")
    parser.add_argument("--max_frames", type=int, default=-1, help="For debugging; -1 means all frames.")
    parser.add_argument("--mode", type=str, default="pad", choices=["pad", "crop"], help="VGGT image preprocessing mode.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional local model.pt path.")
    parser.add_argument("--hf_repo", type=str, default="facebook/VGGT-1B", help="HF repo for from_pretrained fallback.")
    return parser.parse_args()


@torch.no_grad()
def run():
    args = parse_args()
    assert args.chunk_size > 0
    assert 0 <= args.overlap < args.chunk_size

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_paths = list_images(args.image_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {args.image_dir}")
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]

    device = torch.device(args.device)
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8)
        else torch.float16
    )

    # Camera-only model to reduce memory footprint.
    model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        model = VGGT.from_pretrained(args.hf_repo, enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
    model = model.eval().to(device)

    step = args.chunk_size - args.overlap
    frame_to_c2w: Dict[int, torch.Tensor] = {}
    frame_to_name: Dict[int, str] = {}
    frame_to_K: Dict[int, torch.Tensor] = {}
    frame_to_hw: Dict[int, Tuple[int, int]] = {}
    num_frames = len(image_paths)

    print(
        f"[VGGT Prior] Frames={num_frames}, chunk_size={args.chunk_size}, overlap={args.overlap}, step={step}"
    )

    for chunk_start in range(0, num_frames, step):
        chunk_end = min(num_frames, chunk_start + args.chunk_size)
        chunk_ids = list(range(chunk_start, chunk_end))
        chunk_paths = [image_paths[i] for i in chunk_ids]
        for idx in chunk_ids:
            frame_to_name[idx] = os.path.basename(image_paths[idx])

        images = load_and_preprocess_images(chunk_paths, mode=args.mode).to(device)
        with torch.cuda.amp.autocast(dtype=dtype, enabled=(device.type == "cuda")):
            aggregated_tokens, _ = model.aggregator(images[None])
            pose_enc = model.camera_head(aggregated_tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Local chunk camera-from-world -> local c2w map
        extrinsic = extrinsic.squeeze(0)  # [S,3,4]
        intrinsic = intrinsic.squeeze(0)  # [S,3,3]
        proc_h, proc_w = int(images.shape[-2]), int(images.shape[-1])
        local_c2w = {fid: c2w_from_w2c(extrinsic[i]) for i, fid in enumerate(chunk_ids)}
        for i, fid in enumerate(chunk_ids):
            if fid not in frame_to_K:
                frame_to_K[fid] = intrinsic[i].detach().cpu()
                frame_to_hw[fid] = (proc_w, proc_h)

        if len(frame_to_c2w) == 0:
            for fid in chunk_ids:
                frame_to_c2w[fid] = local_c2w[fid].detach().cpu()
        else:
            # Anchor by first overlap frame already present in global map.
            overlap_ids = [fid for fid in chunk_ids if fid in frame_to_c2w]
            if len(overlap_ids) == 0:
                # Fallback: stitch by the earliest global frame.
                anchor_id = min(frame_to_c2w.keys())
            else:
                anchor_id = overlap_ids[0]

            T_global_anchor = frame_to_c2w[anchor_id].to(device)
            T_local_anchor = local_c2w[anchor_id]
            T_global_from_local = T_global_anchor @ torch.linalg.inv(T_local_anchor)

            for fid in chunk_ids:
                T_global = T_global_from_local @ local_c2w[fid]
                if fid not in frame_to_c2w:
                    frame_to_c2w[fid] = T_global.detach().cpu()

        del images, aggregated_tokens, pose_enc, extrinsic, intrinsic
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[VGGT Prior] chunk {chunk_start}:{chunk_end} done.")

    sorted_ids = sorted(frame_to_c2w.keys())
    os.makedirs(os.path.dirname(args.output_tum) or ".", exist_ok=True)
    with open(args.output_tum, "w") as f:
        for fid in sorted_ids:
            T_wc = frame_to_c2w[fid].numpy()
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]
            q = quat_xyzw_from_rot(R_wc)
            f.write(
                f"{float(fid):.6f} {t_wc[0]:.9f} {t_wc[1]:.9f} {t_wc[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )
    print(f"[VGGT Prior] Wrote TUM trajectory: {args.output_tum}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        import csv

        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "image_name", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
            for fid in sorted_ids:
                T_wc = frame_to_c2w[fid].numpy()
                R_wc = T_wc[:3, :3]
                t_wc = T_wc[:3, 3]
                q = quat_xyzw_from_rot(R_wc)
                writer.writerow(
                    [fid, frame_to_name.get(fid, ""), t_wc[0], t_wc[1], t_wc[2], q[0], q[1], q[2], q[3]]
                )
        print(f"[VGGT Prior] Wrote CSV trajectory: {args.output_csv}")

    # Camera parameter export (W2C + K) for downstream pose/camera initialization.
    camera_csv_path = args.output_camera_csv
    if not camera_csv_path:
        base = os.path.splitext(os.path.basename(args.output_tum))[0]
        camera_csv_path = os.path.join(
            os.path.dirname(args.output_tum),
            f"{base}_camera_params.csv",
        )
    os.makedirs(os.path.dirname(camera_csv_path) or ".", exist_ok=True)
    import csv

    with open(camera_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_id",
                "image_name",
                "r11",
                "r12",
                "r13",
                "r21",
                "r22",
                "r23",
                "r31",
                "r32",
                "r33",
                "tx",
                "ty",
                "tz",
                "fx",
                "fy",
                "cx",
                "cy",
                "proc_width",
                "proc_height",
            ]
        )
        for fid in sorted_ids:
            T_wc = frame_to_c2w[fid].numpy()
            T_cw = w2c_from_c2w(T_wc)
            R = T_cw[:3, :3]
            t = T_cw[:3, 3]
            K = frame_to_K[fid].numpy()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            w, h = frame_to_hw[fid]
            writer.writerow(
                [
                    fid,
                    frame_to_name.get(fid, ""),
                    R[0, 0],
                    R[0, 1],
                    R[0, 2],
                    R[1, 0],
                    R[1, 1],
                    R[1, 2],
                    R[2, 0],
                    R[2, 1],
                    R[2, 2],
                    t[0],
                    t[1],
                    t[2],
                    fx,
                    fy,
                    cx,
                    cy,
                    w,
                    h,
                ]
            )
    print(f"[VGGT Prior] Wrote camera params CSV: {camera_csv_path}")

    if args.output_camera_npz:
        os.makedirs(os.path.dirname(args.output_camera_npz) or ".", exist_ok=True)
        ids = np.asarray(sorted_ids, dtype=np.int64)
        names = np.asarray([frame_to_name.get(fid, "") for fid in sorted_ids], dtype=object)
        c2w = np.stack([frame_to_c2w[fid].numpy() for fid in sorted_ids], axis=0)  # [N,4,4]
        w2c = np.stack([w2c_from_c2w(frame_to_c2w[fid].numpy()) for fid in sorted_ids], axis=0)
        K = np.stack([frame_to_K[fid].numpy() for fid in sorted_ids], axis=0)  # [N,3,3]
        hw = np.asarray([frame_to_hw[fid] for fid in sorted_ids], dtype=np.int64)  # [N,2], (w,h)
        np.savez(
            args.output_camera_npz,
            frame_ids=ids,
            image_names=names,
            c2w=c2w,
            w2c=w2c,
            K=K,
            proc_hw=hw,
        )
        print(f"[VGGT Prior] Wrote camera params NPZ: {args.output_camera_npz}")


if __name__ == "__main__":
    run()
