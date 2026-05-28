from __future__ import annotations

import torch

from poses.parallax_geometry import so3_exp
from scene.adaptive_tsdf import AdaptiveTSDF
from scene.anchor_local_map import AnchorLocalMap
from scene.render_guard import RenderGuard


def run_smoke(device: str | torch.device = "cuda") -> dict:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    tsdf = AdaptiveTSDF(device=device)
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
    tsdf.integrate_samples(pts, torch.tensor([0.0, 0.2, 0.0], device=device), torch.ones(3, device=device))
    query = tsdf.query(pts)

    anchor = AnchorLocalMap.create(0, device=device)
    R_new = so3_exp(torch.tensor([0.01, -0.02, 0.03], device=device))
    anchor.update_pose_with_covariance_rotation(R_new, torch.tensor([1.0, 0.0, 0.0], device=device))

    guard = RenderGuard(f=500.0)
    params = {
        "xyz": torch.tensor([[0.0, 0.0, 5.0], [float("nan"), 0.0, 1.0]], device=device),
        "f_dc": torch.zeros(2, 1, 3, device=device),
        "f_rest": torch.zeros(2, 15, 3, device=device),
        "opacity": torch.zeros(2, 1, device=device),
        "scaling": torch.full((2, 3), -3.0, device=device),
        "rotation": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], device=device),
    }
    guard_result = guard.filter(params, torch.zeros(3, device=device))
    return {
        "tsdf_valid": int(query.valid.sum().item()),
        "tsdf_voxels": int(tsdf.keys.shape[0]),
        "anchor_positive_cov": int(anchor.covariance_positive_mask().sum().item()),
        "render_guard_kept": int(guard_result.mask.sum().item()),
    }


if __name__ == "__main__":
    print(run_smoke())
