from __future__ import annotations

from dataclasses import dataclass
from argparse import Namespace

import torch
import torch.nn.functional as F

from dataloaders.read_write_model import BaseImage, Camera, rotmat2qvec
from pipeline.frame_state import FrameState
from poses.feature_detector import DescribedKeypoints
from poses.triangulator import Triangulator
from utils import depth2points, make_torch_sampler, sample, sixD2mtx


def _as_focal_tensor(focal: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(focal):
        f = focal.detach().clone().to(device)
    else:
        f = torch.tensor([float(focal)], dtype=torch.float32, device=device)
    if f.ndim == 0:
        f = f.reshape(1)
    return f.flatten()[:1].contiguous()


def _as_bchw_depth(depth: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if depth.ndim == 2:
        depth = depth[None, None]
    elif depth.ndim == 3:
        depth = depth[None] if depth.shape[0] != 1 else depth[:, None]
    if depth.shape[-2:] != (height, width):
        depth = F.interpolate(depth.float(), (height, width), mode="bilinear", align_corners=True)
    return depth.contiguous()


@dataclass
class TrackingKeyframe:
    """Pipeline-owned keyframe used by pose initialization and anchor-local training."""

    frame: FrameState
    Rt: torch.Tensor
    index: int
    f: torch.Tensor
    centre: torch.Tensor
    triangulator: Triangulator
    args: Namespace

    def __post_init__(self) -> None:
        device = self.Rt.device
        self.info = self.frame.info
        self.desc_kpts: DescribedKeypoints = self.frame.desc_kpts
        self.image_pyr = [self.frame.image.to(device)]
        self.mask_pyr = None
        self.width = self.frame.width
        self.height = self.frame.height
        self.f = _as_focal_tensor(self.f, device)
        self.centre = self.centre.to(device).flatten()[:2].contiguous()
        self.Rt = self.Rt.to(device).float()
        self.rW2C = self.Rt[:3, :2].clone()
        self.tW2C = self.Rt[:3, 3].clone()
        self.approx_centre = self._camera_centre_from_rt(self.Rt)
        self.latest_invdepth = None
        self.depth_scale = torch.ones(1, device=device)
        self.depth_offset = torch.zeros(1, device=device)
        self.exposure = torch.eye(3, 4, device=device)
        self.num_steps = 0
        self.pyr_lvl = int(getattr(self.args, "pyr_levels", 1)) - 1
        self.mono_idepth = _as_bchw_depth(self.frame.mono_idepth.to(device), self.height, self.width)
        self.mono_depth_conf = _as_bchw_depth(self.frame.mono_depth_conf.to(device), self.height, self.width)
        self.feat_map = self.frame.dense_features
        if self.feat_map is not None:
            self.feat_map = self.feat_map.to(device)
        self.idepth_pyr = [self.mono_idepth[0]]
        for _ in range(max(0, int(getattr(self.args, "pyr_levels", 1)) - 1)):
            self.image_pyr.append(F.avg_pool2d(self.image_pyr[-1], 2))
            self.idepth_pyr.append(F.avg_pool2d(self.idepth_pyr[-1], 2))
        if self.frame.mask is not None:
            self.mask_pyr = [self.frame.mask.to(device)]
            if self.mask_pyr[0].ndim == 2:
                self.mask_pyr[0] = self.mask_pyr[0][None]
            for _ in range(max(0, int(getattr(self.args, "pyr_levels", 1)) - 1)):
                self.mask_pyr.append(F.avg_pool2d(self.mask_pyr[-1].float(), 2) > (1 - 1e-6))

    @property
    def lastest_invdepth(self):
        return self.latest_invdepth

    @lastest_invdepth.setter
    def lastest_invdepth(self, value) -> None:
        self.latest_invdepth = value

    @property
    def device(self) -> torch.device:
        return self.tW2C.device

    @property
    def is_test(self) -> bool:
        return bool(self.info.get("is_test", False))

    def to(self, device: str | torch.device, only_train: bool = False) -> None:
        device = torch.device(device)
        self.Rt = self.Rt.to(device)
        self.rW2C = self.rW2C.to(device)
        self.tW2C = self.tW2C.to(device)
        self.f = self.f.to(device)
        self.centre = self.centre.to(device)
        self.approx_centre = self.approx_centre.to(device)
        self.depth_scale = self.depth_scale.to(device)
        self.depth_offset = self.depth_offset.to(device)
        self.exposure = self.exposure.to(device)
        self.image_pyr = [item.to(device) for item in self.image_pyr]
        self.idepth_pyr = [item.to(device) for item in self.idepth_pyr]
        self.mono_idepth = self.mono_idepth.to(device)
        self.mono_depth_conf = self.mono_depth_conf.to(device)
        if self.mask_pyr is not None:
            self.mask_pyr = [item.to(device) for item in self.mask_pyr]
        if not only_train and self.feat_map is not None:
            self.feat_map = self.feat_map.to(device)
        if self.latest_invdepth is not None:
            self.latest_invdepth = self.latest_invdepth.to(device)
        self.desc_kpts.to(device)

    def get_R(self) -> torch.Tensor:
        return sixD2mtx(self.rW2C)

    def get_t(self) -> torch.Tensor:
        return self.tW2C

    def get_Rt(self) -> torch.Tensor:
        Rt = torch.eye(4, dtype=self.tW2C.dtype, device=self.tW2C.device)
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.tW2C
        return Rt

    def set_Rt(self, Rt: torch.Tensor) -> None:
        Rt = Rt.to(self.device).float()
        self.Rt = Rt
        self.rW2C = Rt[:3, :2].clone()
        self.tW2C = Rt[:3, 3].clone()
        self.approx_centre = self._camera_centre_from_rt(Rt)

    def get_centre(self, approx: bool = False) -> torch.Tensor:
        if approx:
            return self.approx_centre
        return self._camera_centre_from_rt(self.get_Rt())

    def get_mono_idepth(self, lvl: int = 0) -> torch.Tensor:
        return self.idepth_pyr[int(lvl)] * self.depth_scale + self.depth_offset

    def apply_mono_idepth_calibration(self, scale: torch.Tensor | float, offset: torch.Tensor | float = 0.0) -> None:
        scale_t = torch.as_tensor(scale, dtype=self.mono_idepth.dtype, device=self.device).reshape(1, 1, 1, 1)
        offset_t = torch.as_tensor(offset, dtype=self.mono_idepth.dtype, device=self.device).reshape(1, 1, 1, 1)
        calibrated = (self.mono_idepth * scale_t + offset_t).clamp_min(1e-6).contiguous()
        self.mono_idepth = calibrated
        self.frame.mono_idepth = calibrated
        self.frame.mono_depth_conf = self.mono_depth_conf
        self.depth_scale = torch.ones(1, device=self.device, dtype=calibrated.dtype)
        self.depth_offset = torch.zeros(1, device=self.device, dtype=calibrated.dtype)
        self.idepth_pyr = [calibrated[0]]
        for _ in range(max(0, int(getattr(self.args, "pyr_levels", 1)) - 1)):
            self.idepth_pyr.append(F.avg_pool2d(self.idepth_pyr[-1], 2))

    @torch.no_grad()
    def sample_conf(self, uv: torch.Tensor) -> torch.Tensor:
        return sample(self.mono_depth_conf, uv.view(1, 1, -1, 2), self.width, self.height)[0, 0, 0]

    @torch.no_grad()
    def update_3dpts(self, all_keyframes: list["TrackingKeyframe"]) -> None:
        unload_desc_kpts = self.desc_kpts.kpts.device.type == "cpu"
        if unload_desc_kpts:
            self.desc_kpts.to(self.device)

        if self.latest_invdepth is not None:
            uv = self.desc_kpts.kpts
            sampler = make_torch_sampler(uv.view(1, 1, -1, 2), self.width, self.height)
            latest = self.latest_invdepth
            if latest.ndim == 2:
                latest = latest[None, None]
            elif latest.ndim == 3:
                latest = latest[None]
            model_idepth = F.grid_sample(latest.to(self.device), sampler, mode="bilinear", align_corners=True)[0, 0, 0]
            mono_idepth = F.grid_sample(self.get_mono_idepth()[None], sampler, mode="bilinear", align_corners=True)[0, 0, 0]
            mono_conf = F.grid_sample(self.mono_depth_conf, sampler, mode="bilinear", align_corners=True)[0, 0, 0]
            conf = 0.1 * torch.exp(-((model_idepth - mono_idepth) ** 2) / 0.2) * mono_conf
            depth = 1 / model_idepth.clamp(1e-6, 1e6)
            new_pts = depth2points(uv, depth[..., None], self.f, self.centre)
            new_pts = (new_pts - self.get_t()) @ self.get_R()
            mask = conf > 0
            self.desc_kpts.update_3D_pts(new_pts[mask], depth[mask], conf[mask], mask)

        uv, uvs_others, chosen_kfs_ids = self.triangulator.prepare_matches(self.desc_kpts)
        id_to_kf = {int(kf.index): kf for kf in all_keyframes}
        Rts_others = []
        for i, kf_id in enumerate(chosen_kfs_ids):
            kf_other = id_to_kf.get(int(kf_id), None)
            if kf_other is None:
                uvs_others[i] = -1
                Rts_others.append(torch.eye(4, device=self.device))
            else:
                Rts_others.append(kf_other.get_Rt().to(self.device))
        if len(Rts_others) == 0:
            Rts = torch.empty(0, 4, 4, device=self.device)
        else:
            Rts = torch.stack(Rts_others, dim=0)
        if len(Rts) < self.triangulator.n_cams:
            pad = torch.eye(4, device=self.device)[None].repeat(self.triangulator.n_cams - len(Rts), 1, 1)
            Rts = torch.cat([Rts, pad], dim=0)

        new_pts, depth, _, valid_matches = self.triangulator(uv, uvs_others, self.get_Rt(), Rts, self.f, self.centre)
        self.desc_kpts.update_3D_pts(new_pts[valid_matches], depth[valid_matches], 1, valid_matches)

        if unload_desc_kpts:
            self.desc_kpts.to("cpu")

    def to_json(self) -> dict:
        info = {"is_test": self.is_test, "frame_id": int(self.frame.frame_id)}
        if "name" in self.info:
            info["name"] = self.info["name"]
        if "Rt" in self.info:
            info["gt_Rt"] = self.info["Rt"].detach().cpu().numpy().tolist()
        if "pose_initialization" in self.info:
            info["pose_initialization"] = dict(self.info["pose_initialization"])
        if "mono_depth_alignment" in self.info:
            info["mono_depth_alignment"] = dict(self.info["mono_depth_alignment"])
        return {"info": info, "Rt": self.get_Rt().detach().cpu().numpy().tolist(), "f": float(self.f.item())}

    def to_colmap(self, id: int) -> tuple[Camera, BaseImage]:
        camera = Camera(
            id=id,
            model="SIMPLE_PINHOLE",
            width=self.width,
            height=self.height,
            params=[float(self.f.item()), float(self.centre[0].item()), float(self.centre[1].item())],
        )
        image = BaseImage(
            id=id,
            name=self.info.get("name", str(id)),
            camera_id=id,
            qvec=-rotmat2qvec(self.get_R().detach().cpu().numpy()),
            tvec=self.get_t().flatten().detach().cpu().numpy(),
            xys=[],
            point3D_ids=[],
        )
        return camera, image

    @staticmethod
    def _camera_centre_from_rt(Rt: torch.Tensor) -> torch.Tensor:
        return -Rt[:3, :3].T @ Rt[:3, 3]


class TrackingKeyframeStore:
    """Owns progressive keyframes and previous-keyframe selection."""

    def __init__(
        self,
        width: int,
        height: int,
        args: Namespace,
        matcher,
        triangulator: Triangulator,
        device: str | torch.device = "cuda",
    ):
        self.width = int(width)
        self.height = int(height)
        self.args = args
        self.matcher = matcher
        self.triangulator = triangulator
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], dtype=torch.float32, device=self.device)
        self.f = float(0.7 * width)
        self.current_lod = int(getattr(args, "lod_min", 1))
        self.num_prev_keyframes_check = int(getattr(args, "num_prev_keyframes_check", 10))
        self.keyframes: list[TrackingKeyframe] = []
        self.frame_to_keyframe: dict[int, TrackingKeyframe] = {}
        self.approx_cam_centres: torch.Tensor | None = None
        self.sorted_frame_indices = torch.empty(0, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.keyframes)

    def add(self, frame: FrameState, Rt: torch.Tensor, focal: torch.Tensor | float, index: int) -> TrackingKeyframe:
        self.f = float(_as_focal_tensor(focal, Rt.device if torch.is_tensor(Rt) else self.device).item())
        keyframe = TrackingKeyframe(
            frame=frame,
            Rt=Rt,
            index=int(index),
            f=torch.tensor([self.f], device=Rt.device if torch.is_tensor(Rt) else self.device),
            centre=self.centre,
            triangulator=self.triangulator,
            args=self.args,
        )
        self.keyframes.append(keyframe)
        self.frame_to_keyframe[int(frame.frame_id)] = keyframe
        self._refresh_sort(keyframe.approx_centre)
        return keyframe

    def add_keyframe(self, frame: FrameState, Rt: torch.Tensor, focal: torch.Tensor | float, index: int) -> TrackingKeyframe:
        return self.add(frame, Rt, focal, index)

    def by_index(self, index: int) -> TrackingKeyframe | None:
        if 0 <= int(index) < len(self.keyframes):
            return self.keyframes[int(index)]
        return None

    def by_frame_id(self, frame_id: int) -> TrackingKeyframe | None:
        return self.frame_to_keyframe.get(int(frame_id), None)

    def recent(self, n: int) -> list[TrackingKeyframe]:
        return self.keyframes[-int(n):]

    @torch.no_grad()
    def get_prev_keyframes(
        self,
        n: int,
        update_3dpts: bool,
        desc_kpts: DescribedKeypoints | None = None,
    ) -> list[TrackingKeyframe]:
        n = min(int(n), len(self.keyframes))
        if n <= 0:
            return []
        if desc_kpts is not None and len(self.keyframes) > n:
            n_checks = min(self.num_prev_keyframes_check, len(self.keyframes))
            indices_to_check = self.sorted_frame_indices[:n_checks]
            scores = torch.zeros(len(indices_to_check), device=self.device)
            for i, index in enumerate(indices_to_check):
                scores[i] = self.matcher.evaluate_match(self.keyframes[int(index)].desc_kpts, desc_kpts)
            _, top_indices = torch.topk(scores, n)
            selected = indices_to_check[top_indices.cpu()]
        else:
            selected = self.sorted_frame_indices[:n]
        prev_keyframes = [self.keyframes[int(i)] for i in selected]
        if update_3dpts:
            for keyframe in prev_keyframes:
                keyframe.update_3dpts(self.keyframes)
        return prev_keyframes

    def refresh_after_pose_updates(self) -> None:
        if len(self.keyframes) == 0:
            self.approx_cam_centres = None
            self.sorted_frame_indices = torch.empty(0, dtype=torch.long)
            return
        centers = [kf.get_centre(approx=True).detach().to(self.device) for kf in self.keyframes]
        self.approx_cam_centres = torch.stack(centers, dim=0)
        self._refresh_sort(self.approx_cam_centres[-1])

    def _refresh_sort(self, reference_centre: torch.Tensor) -> None:
        centers = [kf.get_centre(approx=True).detach().to(self.device) for kf in self.keyframes]
        self.approx_cam_centres = torch.stack(centers, dim=0)
        dist = torch.linalg.vector_norm(self.approx_cam_centres - reference_centre.to(self.device)[None], dim=-1)
        self.sorted_frame_indices = torch.argsort(dist).cpu()
