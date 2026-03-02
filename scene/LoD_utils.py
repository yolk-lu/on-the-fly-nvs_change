import math
from typing import Sequence

import torch


def _extract_xyz(gaussians):
    if isinstance(gaussians, dict):
        if "xyz" in gaussians and isinstance(gaussians["xyz"], dict):
            return gaussians["xyz"]["val"]
        if "xyz" in gaussians:
            return gaussians["xyz"]
    if hasattr(gaussians, "xyz"):
        return gaussians.xyz
    raise ValueError("Unable to extract xyz from gaussians input")


def _extract_scaling(gaussians):
    if isinstance(gaussians, dict):
        if "scaling" in gaussians and isinstance(gaussians["scaling"], dict):
            return gaussians["scaling"]["val"]
        if "scaling" in gaussians:
            return gaussians["scaling"]
    if hasattr(gaussians, "gaussian_params") and "scaling" in gaussians.gaussian_params:
        return gaussians.gaussian_params["scaling"]["val"]
    if hasattr(gaussians, "scaling"):
        return gaussians.scaling
    raise ValueError("Unable to extract scaling from gaussians input")


def get_gaussians_distance(gaussians, camera_position):
    xyz = _extract_xyz(gaussians)
    if camera_position.dim() == 1:
        camera_position = camera_position[None]
    return torch.linalg.norm(xyz - camera_position.to(xyz.device), dim=-1)


def get_scale(gaussians, scaling_lower_bound: float = 0.0):
    scaling = _extract_scaling(gaussians)
    return torch.exp(scaling) + float(scaling_lower_bound)


def get_gaussians_radius(gaussians, scaling_lower_bound: float = 0.0):
    scale = get_scale(gaussians, scaling_lower_bound=scaling_lower_bound)
    return scale.max(dim=-1).values


def level_of_gaussians(
    gaussians,
    camera_position,
    num_levels: int = 3,
    quantiles: Sequence[float] | None = None,
):
    """
    Split gaussians into LoD groups by camera distance.
    Level 0 = nearest (high LoD), Level num_levels-1 = farthest (low LoD).
    """
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")

    d = get_gaussians_distance(gaussians, camera_position).detach()
    if num_levels == 1 or d.numel() == 0:
        return {0: torch.arange(d.shape[0], device=d.device)}

    if quantiles is None:
        quantiles = [i / num_levels for i in range(1, num_levels)]
    if len(quantiles) != num_levels - 1:
        raise ValueError("quantiles length must be num_levels - 1")

    q = torch.tensor(quantiles, device=d.device, dtype=d.dtype).clamp(0.0, 1.0)
    thresholds = torch.quantile(d, q)

    levels = {}
    prev = torch.full_like(d, True, dtype=torch.bool)
    for i, thr in enumerate(thresholds):
        mask = d <= thr
        levels[i] = torch.where(prev & mask)[0]
        prev = prev & (~mask)
    levels[num_levels - 1] = torch.where(prev)[0]
    return levels


def progressive_merging_gaussians(
    activated_anchors,
    gaussians,
    camera_position,
    scale: float = 1.0,
    temperature: float = 1.0,
):
    """
    Compute anchor blending weights for progressive training/rendering.
    Returns a weight tensor aligned with activated_anchors.
    """
    if activated_anchors is None or len(activated_anchors) == 0:
        return torch.ones(1, device=camera_position.device)

    anchor_positions = torch.stack([anchor.position for anchor in activated_anchors]).to(camera_position.device)
    d = torch.linalg.norm(anchor_positions - camera_position[None], dim=-1)
    d = d / max(float(scale), 1e-6)
    logits = -d / max(float(temperature), 1e-6)
    return torch.softmax(logits, dim=0)


def combine_different_levels(level_groups, keep_levels=None):
    if keep_levels is None:
        keep_levels = sorted(level_groups.keys())
    selected = [level_groups[lvl] for lvl in keep_levels if lvl in level_groups]
    if len(selected) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.cat(selected, dim=0), sorted=True)


def high_lod(level_groups):
    if len(level_groups) == 0:
        return torch.empty(0, dtype=torch.long)
    return level_groups[min(level_groups.keys())]


def low_lod(level_groups):
    if len(level_groups) == 0:
        return torch.empty(0, dtype=torch.long)
    return level_groups[max(level_groups.keys())]


def merge_lod(gaussians, indices):
    if indices is None:
        return gaussians
    merged = {}
    for key, value in gaussians.items():
        if isinstance(value, dict) and "val" in value:
            merged[key] = {k: v for k, v in value.items()}
            merged[key]["val"] = value["val"][indices]
        elif torch.is_tensor(value):
            merged[key] = value[indices]
        else:
            merged[key] = value
    return merged


def projection_screen_error(
    gaussians,
    view_matrix,
    focal,
    image_width,
    image_height,
    scaling_lower_bound: float = 0.0,
):
    """
    Approximate projection error in pixels from Gaussian screen footprint.
    """
    xyz = _extract_xyz(gaussians)
    radius = get_gaussians_radius(gaussians, scaling_lower_bound=scaling_lower_bound)

    R = view_matrix[:3, :3]
    t = view_matrix[:3, 3]
    xyz_cam = (R @ xyz.T).T + t[None]
    z = xyz_cam[:, 2].clamp(min=1e-6)

    focal_px = float(focal)
    if focal_px <= 0:
        focal_px = 0.5 * (image_width + image_height)

    err_px = focal_px * radius / z
    err_px = torch.nan_to_num(err_px, nan=0.0, posinf=1e6, neginf=0.0)
    return {
        "mean": err_px.mean().item() if err_px.numel() > 0 else 0.0,
        "median": err_px.median().item() if err_px.numel() > 0 else 0.0,
        "p90": torch.quantile(err_px, 0.90).item() if err_px.numel() > 0 else 0.0,
        "max": err_px.max().item() if err_px.numel() > 0 else 0.0,
    }


def pose_stability_score(keyframes, window: int = 8):
    """
    Estimate global pose stability in [0, 1].
    1 means more stable camera trajectory in recent frames.
    """
    if keyframes is None or len(keyframes) < 3:
        return 0.0

    recent = keyframes[-max(3, window):]
    centres = torch.stack([kf.get_centre(approx=True).detach() for kf in recent])
    disp = torch.linalg.norm(centres[1:] - centres[:-1], dim=-1)
    mean_disp = disp.mean().item() + 1e-6
    cv_disp = (disp.std().item() / mean_disp) if disp.numel() > 1 else 0.0

    rts = torch.stack([kf.get_Rt().detach() for kf in recent])
    Rs = rts[:, :3, :3]
    rel = Rs[1:] @ Rs[:-1].transpose(-1, -2)
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    rot_rad = torch.arccos(cos_theta)
    rot_mean = rot_rad.mean().item() if rot_rad.numel() > 0 else 0.0

    score = math.exp(-1.5 * cv_disp) * math.exp(-2.0 * rot_mean)
    return float(max(0.0, min(1.0, score)))


def lod_progressive_ready(
    scene_model,
    min_pose_stability: float = 0.55,
    max_projection_error_px: float = 3.0,
    pose_window: int = 8,
):
    if len(scene_model.keyframes) == 0:
        return {
            "ready": False,
            "pose_stability": 0.0,
            "projection_error_mean": float("inf"),
            "reason": "no_keyframes",
        }

    pose_score = pose_stability_score(scene_model.keyframes, window=pose_window)
    last_kf = scene_model.keyframes[-1]
    proj = projection_screen_error(
        scene_model.active_anchor.gaussian_params,
        last_kf.get_Rt().detach(),
        scene_model.f,
        scene_model.width,
        scene_model.height,
        scaling_lower_bound=getattr(scene_model, "scaling_lower_bound", 0.0),
    )

    ready = pose_score >= float(min_pose_stability) and proj["mean"] <= float(max_projection_error_px)
    reason = "ok" if ready else (
        "unstable_pose" if pose_score < float(min_pose_stability) else "high_projection_error"
    )
    return {
        "ready": ready,
        "pose_stability": pose_score,
        "projection_error_mean": proj["mean"],
        "projection_error_p90": proj["p90"],
        "reason": reason,
    }