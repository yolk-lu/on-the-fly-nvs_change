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


def _extract_rotation(gaussians):
    if isinstance(gaussians, dict):
        if "rotation" in gaussians and isinstance(gaussians["rotation"], dict):
            return gaussians["rotation"]["val"]
        if "rotation" in gaussians:
            return gaussians["rotation"]
    if hasattr(gaussians, "gaussian_params") and "rotation" in gaussians.gaussian_params:
        return gaussians.gaussian_params["rotation"]["val"]
    if hasattr(gaussians, "rotation"):
        return gaussians.rotation
    return None


def _quaternion_to_rotation_matrix(q):
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    row0 = torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def projected_major_axis_px(
    gaussians,
    view_matrix,
    focal,
    image_width,
    image_height,
    scaling_lower_bound: float = 0.0,
):
    """
    View-dependent projected major-axis length (px) per Gaussian.

    Uses the covariance projection form:
      Sigma_img = J(v) W(v) Sigma W(v)^T J(v)^T
    and returns sqrt(lambda_max(Sigma_img)).
    """
    xyz = _extract_xyz(gaussians)
    scale = get_scale(gaussians, scaling_lower_bound=scaling_lower_bound)
    rot_q = _extract_rotation(gaussians)

    Rcw = view_matrix[:3, :3].to(xyz.device)
    tcw = view_matrix[:3, 3].to(xyz.device)
    xyz_cam = (Rcw @ xyz.T).T + tcw[None]
    z = xyz_cam[:, 2].clamp(min=1e-6)

    focal_px = float(focal)
    if focal_px <= 0:
        focal_px = 0.5 * (image_width + image_height)
    fx = torch.full_like(z, focal_px)
    fy = torch.full_like(z, focal_px)

    if rot_q is None:
        radius = scale.max(dim=-1).values
        err_px = focal_px * radius / z
        return torch.nan_to_num(err_px, nan=0.0, posinf=1e6, neginf=0.0)

    Robj = _quaternion_to_rotation_matrix(rot_q.to(xyz.device))
    D = torch.diag_embed(scale.square())
    sigma_world = Robj @ D @ Robj.transpose(-1, -2)

    Rcw_expand = Rcw[None].expand(sigma_world.shape[0], -1, -1)
    sigma_cam = Rcw_expand @ sigma_world @ Rcw_expand.transpose(-1, -2)

    x = xyz_cam[:, 0]
    y = xyz_cam[:, 1]
    J = torch.zeros((xyz_cam.shape[0], 2, 3), device=xyz.device, dtype=xyz.dtype)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / (z * z)

    sigma_img = J @ sigma_cam @ J.transpose(-1, -2)
    sigma_img = 0.5 * (sigma_img + sigma_img.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(sigma_img)
    major_axis = torch.sqrt(torch.clamp(eigvals[:, -1], min=0.0) + 1e-8)
    return torch.nan_to_num(major_axis, nan=0.0, posinf=1e6, neginf=0.0)


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


def distance_level_colors(
    gaussians,
    camera_position,
    num_levels: int = 4,
    palette: torch.Tensor | None = None,
):
    """
    Assign each gaussian a distance-based LoD level and RGB debug color.

    Returns:
      level_ids: LongTensor [N]
      colors: FloatTensor [N,3] in [0,1]
    """
    xyz = _extract_xyz(gaussians)
    n = xyz.shape[0]
    if n == 0:
        return (
            torch.empty(0, dtype=torch.long, device=xyz.device),
            torch.empty(0, 3, dtype=xyz.dtype, device=xyz.device),
        )

    groups = level_of_gaussians(gaussians, camera_position, num_levels=num_levels)
    level_ids = torch.zeros(n, dtype=torch.long, device=xyz.device)
    for lvl, idx in groups.items():
        level_ids[idx] = int(lvl)

    if palette is None:
        palette = torch.tensor(
            [
                [1.0, 0.1, 0.1],
                [1.0, 0.6, 0.1],
                [1.0, 1.0, 0.1],
                [0.1, 1.0, 0.1],
                [0.1, 1.0, 1.0],
                [0.1, 0.4, 1.0],
                [0.8, 0.1, 1.0],
            ],
            dtype=xyz.dtype,
            device=xyz.device,
        )
    else:
        palette = palette.to(device=xyz.device, dtype=xyz.dtype)

    if num_levels <= palette.shape[0]:
        level_palette = palette[:num_levels]
    else:
        sample_ids = torch.linspace(0, palette.shape[0] - 1, steps=num_levels, device=xyz.device)
        sample_ids = torch.round(sample_ids).long().clamp(0, palette.shape[0] - 1)
        level_palette = palette[sample_ids]

    colors = level_palette[level_ids]
    return level_ids, colors


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
    err_px = projected_major_axis_px(
        gaussians,
        view_matrix,
        focal,
        image_width,
        image_height,
        scaling_lower_bound=scaling_lower_bound,
    )
    return {
        "mean": err_px.mean().item() if err_px.numel() > 0 else 0.0,
        "median": err_px.median().item() if err_px.numel() > 0 else 0.0,
        "p90": torch.quantile(err_px, 0.90).item() if err_px.numel() > 0 else 0.0,
        "max": err_px.max().item() if err_px.numel() > 0 else 0.0,
    }


def distance_boundary_penalty(
    base_penalty: torch.Tensor,
    uv_grid: torch.Tensor,
    width: int,
    height: int,
    rendered_depth: torch.Tensor | None = None,
    boundary_weight: float = 0.35,
    distance_weight: float = 0.20,
    boundary_margin_ratio: float = 0.08,
):
    """
    Build a spatial penalty map that mixes:
    - boundary penalty near image borders
    - depth-distance penalty (farther pixels penalized more)

    Returns a tensor with the same shape as base_penalty.
    """
    penalty = base_penalty
    device = penalty.device
    dtype = penalty.dtype

    if boundary_weight > 0:
        margin_ratio = float(max(0.0, min(0.5, boundary_margin_ratio)))
        margin_px = int(min(width, height) * margin_ratio)
        if margin_px > 0:
            x = uv_grid[..., 0]
            y = uv_grid[..., 1]
            dist_x = torch.minimum(x, (width - 1) - x)
            dist_y = torch.minimum(y, (height - 1) - y)
            dist_to_edge = torch.minimum(dist_x, dist_y)
            border_map = ((margin_px - dist_to_edge).clamp(min=0.0) / float(margin_px)).clamp(0.0, 1.0)
            penalty = penalty + float(boundary_weight) * border_map.to(device=device, dtype=dtype)

    if rendered_depth is not None and distance_weight > 0:
        depth = rendered_depth.to(device=device, dtype=dtype)
        valid = torch.isfinite(depth) & (depth > 0)
        if valid.any():
            d_valid = depth[valid]
            d_lo = torch.quantile(d_valid, 0.05)
            d_hi = torch.quantile(d_valid, 0.95)
            if (d_hi - d_lo).abs() < 1e-8:
                depth_norm = torch.zeros_like(depth)
            else:
                depth_norm = ((depth - d_lo) / (d_hi - d_lo)).clamp(0.0, 1.0)
            depth_norm = torch.where(valid, depth_norm, torch.zeros_like(depth_norm))
            penalty = penalty + float(distance_weight) * depth_norm

    return penalty


def gaussian_aspect_ratio_mask(
    log_scaling: torch.Tensor,
    max_aspect_ratio: float,
    scaling_lower_bound: float = 0.0,
):
    """
    Keep gaussians whose anisotropy is below threshold:
      aspect_ratio = max(scale_xyz) / min(scale_xyz)
    """
    if max_aspect_ratio <= 1.0:
        return torch.ones(log_scaling.shape[0], dtype=torch.bool, device=log_scaling.device)

    scales = torch.exp(log_scaling) + float(scaling_lower_bound)
    min_scale = scales.min(dim=-1).values.clamp_min(1e-8)
    max_scale = scales.max(dim=-1).values
    ratio = max_scale / min_scale
    return ratio <= float(max_aspect_ratio)


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
def get_lod_density(
    depth_map: torch.Tensor,
    near_dist: float = 2.0,
    far_dist: float = 15.0,
    far_density_ratio: float = 0.2,
):
    """
    Returns (density_multiplier) masks based on depth.
    Near distances return 1.0. Far distances return far_density_ratio.
    """
    t = ((depth_map - near_dist) / max(1e-5, far_dist - near_dist)).clamp(0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    
    density_multiplier = 1.0 - t * (1.0 - far_density_ratio)
    
    return density_multiplier
def get_lod_merge_k(
    distances: torch.Tensor,
    near_dist: float = 5.0,
    far_dist: float = 15.0,
    k_near: int = 3,
    k_far: int = 7,
):
    """
    Returns an aggressive merge factor K based on distance to camera.
    """
    t = ((distances - near_dist) / max(1e-5, far_dist - near_dist)).clamp(0.0, 1.0)
    k_float = k_near + t * (k_far - k_near)
    return k_float.round().long()

def aspect_ratio_penalty(
    log_scaling: torch.Tensor,
    max_aspect_ratio: float = 8.0,
    scaling_lower_bound: float = 0.0,
):
    """
    L1 penalty on the ratio of max_scale / min_scale if it exceeds max_aspect_ratio.
    Operating directly in log-space guarantees gradients are precisely +/- 1, 
    completely preventing float32 Adam overflows from exp() operations.
    """
    import math
    max_log = log_scaling.max(dim=-1).values
    min_log = log_scaling.min(dim=-1).values
    max_diff = math.log(float(max_aspect_ratio))
    
    penalty = torch.nn.functional.relu(max_log - min_log - max_diff)
    return penalty.mean()
