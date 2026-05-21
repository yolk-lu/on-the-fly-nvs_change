from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.anchor_local_map import AnchorLocalMap
from scene.local_gaussian_model import GAUSSIAN_KEYS
from scene.render_guard import RenderGuard
from utils import focal2fov, getProjectionMatrix


@dataclass
class AnchorRenderBatch:
    params: dict[str, torch.Tensor]
    anchor_ids: torch.Tensor
    local_indices: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.params["xyz"].shape[0])


@dataclass
class AnchorRenderResult:
    render: torch.Tensor
    invdepth: torch.Tensor
    main_gaussian_id: torch.Tensor
    radii: torch.Tensor
    visibility_filter: torch.Tensor
    screenspace_points: torch.Tensor
    anchor_ids: torch.Tensor
    local_indices: torch.Tensor
    kept_indices: torch.Tensor
    guard_reason_counts: dict[str, int]
    raster_limit_count: int = 0
    cuda_error: str = ""


class AnchorLocalRenderer:
    """Anchor-local renderer with explicit pre-rasterizer validation and debug ownership."""

    def __init__(
        self,
        width: int,
        height: int,
        f: torch.Tensor | float,
        sh_degree: int = 3,
        max_screen_px: float = 96.0,
        max_depth: float = 1e4,
        max_rasterized_gaussians: int = 60_000,
        device: str | torch.device = "cuda",
    ):
        self.width = int(width)
        self.height = int(height)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.sh_degree = int(sh_degree)
        self.max_screen_px = float(max_screen_px)
        self.max_depth = float(max_depth)
        self.max_rasterized_gaussians = int(max_rasterized_gaussians)
        self.z_near = 0.01
        self.update_intrinsics(f)

    def update_intrinsics(self, f: torch.Tensor | float) -> None:
        self.f = float(f.detach().flatten()[0].item() if torch.is_tensor(f) else f)
        self.fov_x = focal2fov(self.f, self.width)
        self.fov_y = focal2fov(self.f, self.height)
        self.tanfovx = math.tan(self.fov_x * 0.5)
        self.tanfovy = math.tan(self.fov_y * 0.5)
        self.projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.fov_x, fovY=self.fov_y)
            .transpose(0, 1)
            .to(self.device)
        )
        self.guard = RenderGuard(self.f, max_screen_px=self.max_screen_px, max_depth=self.max_depth)

    def collect_anchor_batch(self, anchors: list[AnchorLocalMap]) -> AnchorRenderBatch:
        non_empty = [anchor for anchor in anchors if anchor.gaussian_model.n > 0]
        if len(non_empty) == 0:
            return self._empty_batch()

        parts: dict[str, list[torch.Tensor]] = {key: [] for key in GAUSSIAN_KEYS}
        anchor_ids = []
        local_indices = []
        for anchor in non_empty:
            world_params = anchor.gaussian_model.world_params(
                anchor.R_anchor_to_world,
                anchor.t_anchor_to_world,
                anchor.s_anchor_to_world,
            )
            n = int(world_params["xyz"].shape[0])
            for key in GAUSSIAN_KEYS:
                parts[key].append(world_params[key].to(self.device))
            anchor_ids.append(torch.full((n,), anchor.anchor_id, device=self.device, dtype=torch.long))
            local_indices.append(torch.arange(n, device=self.device, dtype=torch.long))

        return AnchorRenderBatch(
            params={key: torch.cat(value, dim=0).contiguous() for key, value in parts.items()},
            anchor_ids=torch.cat(anchor_ids, dim=0).contiguous(),
            local_indices=torch.cat(local_indices, dim=0).contiguous(),
        )

    def render(
        self,
        anchors: list[AnchorLocalMap],
        view_matrix: torch.Tensor,
        bg: torch.Tensor | None = None,
        scaling_modifier: float = 1.0,
    ) -> AnchorRenderResult:
        if bg is None:
            bg = torch.zeros(3, device=self.device)
        view_matrix = view_matrix.to(self.device).contiguous()
        cam_centre = view_matrix.detach().inverse()[3, :3]
        batch = self.collect_anchor_batch(anchors)
        if batch.n == 0:
            return self._empty_result(batch, bg.device)

        guard_result = self.guard.filter(batch.params, cam_centre)
        kept = torch.nonzero(guard_result.mask, as_tuple=False).flatten()
        # RenderGuard intentionally runs without autograd. Re-slice the original
        # differentiable batch tensors with the guard mask so gradients still
        # flow back to AnchorLocalMap.gaussian_model.params.
        kept_params = {key: value[kept].contiguous() for key, value in batch.params.items()}
        kept_batch = AnchorRenderBatch(
            params=kept_params,
            anchor_ids=batch.anchor_ids[kept],
            local_indices=batch.local_indices[kept],
        )
        if kept_batch.n == 0:
            return self._empty_result(kept_batch, bg.device, guard_result.reason_counts)
        camera_mask, camera_counts = self._camera_space_filter(kept_batch.params, view_matrix)
        guard_counts = dict(guard_result.reason_counts)
        guard_counts.update(camera_counts)
        if not camera_mask.all():
            camera_kept = torch.nonzero(camera_mask, as_tuple=False).flatten()
            kept = kept[camera_kept]
            kept_batch = AnchorRenderBatch(
                params={key: value[camera_kept].contiguous() for key, value in kept_batch.params.items()},
                anchor_ids=kept_batch.anchor_ids[camera_kept],
                local_indices=kept_batch.local_indices[camera_kept],
            )
            if kept_batch.n == 0:
                return self._empty_result(kept_batch, bg.device, guard_counts)
        raster_limit_count = 0
        if self.max_rasterized_gaussians > 0 and kept_batch.n > self.max_rasterized_gaussians:
            selected = self._select_raster_subset(kept_batch.params, cam_centre, self.max_rasterized_gaussians)
            raster_limit_count = int(kept_batch.n - selected.shape[0])
            kept = kept[selected]
            kept_batch = AnchorRenderBatch(
                params={key: value[selected].contiguous() for key, value in kept_batch.params.items()},
                anchor_ids=kept_batch.anchor_ids[selected],
                local_indices=kept_batch.local_indices[selected],
            )

        params = self._sanitize_raster_params(kept_batch.params)
        screenspace_points = torch.zeros_like(params["xyz"], requires_grad=True)
        raster_settings = GaussianRasterizationSettings(
            self.height,
            self.width,
            self.tanfovx,
            self.tanfovy,
            bg.to(self.device),
            float(scaling_modifier),
            self.projection_matrix,
            self.sh_degree,
            cam_centre,
            False,
            False,
        )
        rasterizer = GaussianRasterizer(raster_settings)
        try:
            color, invdepth, main_gaussian_id, radii = rasterizer(
                params["xyz"].contiguous(),
                screenspace_points.contiguous(),
                params["opacity"].sigmoid().contiguous(),
                params["f_dc"].contiguous(),
                params["f_rest"].contiguous(),
                torch.exp(params["scaling"].clamp(-20.0, 20.0)).contiguous(),
                params["rotation"].contiguous(),
                view_matrix,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            visibility_filter = radii > 0
            cuda_error = ""
        except RuntimeError as exc:
            if "CUDA" not in str(exc):
                raise
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except RuntimeError:
                pass
            color = torch.zeros(3, self.height, self.width, device=self.device)
            invdepth = torch.zeros(1, self.height, self.width, device=self.device)
            main_gaussian_id = torch.zeros(1, self.height, self.width, device=self.device, dtype=torch.int32)
            radii = torch.zeros(kept_batch.n, device=self.device, dtype=torch.int32)
            visibility_filter = torch.zeros(kept_batch.n, device=self.device, dtype=torch.bool)
            cuda_error = f"{type(exc).__name__}: {exc}"

        return AnchorRenderResult(
            render=color,
            invdepth=invdepth,
            main_gaussian_id=main_gaussian_id,
            radii=radii,
            visibility_filter=visibility_filter,
            screenspace_points=screenspace_points,
            anchor_ids=kept_batch.anchor_ids,
            local_indices=kept_batch.local_indices,
            kept_indices=kept,
            guard_reason_counts=guard_counts,
            raster_limit_count=raster_limit_count,
            cuda_error=cuda_error,
        )

    def _empty_batch(self) -> AnchorRenderBatch:
        rest_dim = (self.sh_degree + 1) * (self.sh_degree + 1) - 1
        params = {
            "xyz": torch.empty(0, 3, device=self.device),
            "f_dc": torch.empty(0, 1, 3, device=self.device),
            "f_rest": torch.empty(0, rest_dim, 3, device=self.device),
            "opacity": torch.empty(0, 1, device=self.device),
            "scaling": torch.empty(0, 3, device=self.device),
            "rotation": torch.empty(0, 4, device=self.device),
        }
        return AnchorRenderBatch(
            params=params,
            anchor_ids=torch.empty(0, dtype=torch.long, device=self.device),
            local_indices=torch.empty(0, dtype=torch.long, device=self.device),
        )

    def _empty_result(
        self,
        batch: AnchorRenderBatch,
        device: torch.device,
        reason_counts: dict[str, int] | None = None,
    ) -> AnchorRenderResult:
        return AnchorRenderResult(
            render=torch.zeros(3, self.height, self.width, device=device),
            invdepth=torch.zeros(1, self.height, self.width, device=device),
            main_gaussian_id=torch.zeros(1, self.height, self.width, device=device, dtype=torch.int32),
            radii=torch.zeros(batch.n, device=device, dtype=torch.int32),
            visibility_filter=torch.zeros(batch.n, device=device, dtype=torch.bool),
            screenspace_points=torch.zeros(batch.n, 3, device=device),
            anchor_ids=batch.anchor_ids,
            local_indices=batch.local_indices,
            kept_indices=torch.empty(0, dtype=torch.long, device=device),
            guard_reason_counts={} if reason_counts is None else reason_counts,
            raster_limit_count=0,
        )

    def _select_raster_subset(self, params: dict[str, torch.Tensor], cam_centre: torch.Tensor, max_count: int) -> torch.Tensor:
        with torch.no_grad():
            xyz = params["xyz"]
            dist = torch.linalg.vector_norm(xyz - cam_centre[None], dim=-1).clamp_min(1e-6)
            opacity = params["opacity"].sigmoid().flatten()
            scaling = torch.exp(params["scaling"].clamp(-20.0, 20.0)).max(dim=-1).values
            screen_score = self.f * scaling / dist
            score = opacity * screen_score.clamp_min(1e-8)
            finite = torch.isfinite(score)
            score = torch.where(finite, score, torch.zeros_like(score))
            return torch.topk(score, k=min(int(max_count), int(score.shape[0])), largest=True).indices

    @torch.no_grad()
    def _camera_space_filter(
        self,
        params: dict[str, torch.Tensor],
        view_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        xyz = params["xyz"]
        n = int(xyz.shape[0])
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=xyz.device), {
                "bad_camera_depth": 0,
                "bad_frustum": 0,
                "bad_camera_screen": 0,
            }
        xyz_cam = xyz @ view_matrix[:3, :3] + view_matrix[3:4, :3]
        z = xyz_cam[:, 2]
        finite_cam = torch.isfinite(xyz_cam).all(dim=-1)
        valid_depth = finite_cam & torch.isfinite(z) & (z > self.z_near) & (z < self.max_depth)
        z_safe = z.clamp_min(self.z_near)
        u = self.f * (xyz_cam[:, 0] / z_safe) + (self.width - 1) * 0.5
        v = self.f * (xyz_cam[:, 1] / z_safe) + (self.height - 1) * 0.5
        margin = max(float(self.width), float(self.height)) * 0.25 + self.max_screen_px
        valid_frustum = (
            torch.isfinite(u)
            & torch.isfinite(v)
            & (u >= -margin)
            & (u <= (self.width - 1) + margin)
            & (v >= -margin)
            & (v <= (self.height - 1) + margin)
        )
        scaling = torch.exp(params["scaling"].clamp(-20.0, 20.0)).max(dim=-1).values
        screen = self.f * scaling / z_safe
        valid_screen = torch.isfinite(screen) & (screen > 0) & (screen < self.max_screen_px)
        valid = valid_depth & valid_frustum & valid_screen
        return valid, {
            "bad_camera_depth": int((~valid_depth).sum().item()),
            "bad_frustum": int((valid_depth & ~valid_frustum).sum().item()),
            "bad_camera_screen": int((valid_depth & valid_frustum & ~valid_screen).sum().item()),
        }

    @staticmethod
    def _sanitize_raster_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raw_rot = params["rotation"].contiguous()
        rot_norm = torch.linalg.vector_norm(raw_rot, dim=-1, keepdim=True)
        identity = torch.zeros_like(raw_rot)
        if identity.numel() > 0:
            identity[:, 0] = 1
        rotation = torch.where(rot_norm > 1e-8, raw_rot / rot_norm.clamp_min(1e-8), identity)
        out = {key: value for key, value in params.items()}
        out["rotation"] = rotation.contiguous()
        return out
