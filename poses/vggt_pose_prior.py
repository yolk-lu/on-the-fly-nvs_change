import csv
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def _quat_xyzw_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> torch.Tensor:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return torch.eye(3, dtype=torch.float32)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    R = torch.tensor(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=torch.float32,
    )
    return R


def _tum_line_to_w2c(tokens) -> Optional[tuple[int, torch.Tensor]]:
    if len(tokens) < 8:
        return None
    try:
        ts = float(tokens[0])
        tx, ty, tz = float(tokens[1]), float(tokens[2]), float(tokens[3])
        qx, qy, qz, qw = float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7])
    except ValueError:
        return None

    frame_id = int(round(ts))
    R_wc = _quat_xyzw_to_rotmat(qx, qy, qz, qw)
    t_wc = torch.tensor([tx, ty, tz], dtype=torch.float32)

    # Convert c2w (R_wc, t_wc) -> w2c (R_cw, t_cw)
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    Rt = torch.eye(4, dtype=torch.float32)
    Rt[:3, :3] = R_cw
    Rt[:3, 3] = t_cw
    return frame_id, Rt


class VGGTPosePrior:
    """
    Loads external pose priors (default: TUM trajectory format).
    Stores world-to-camera Rt by frame index.
    """

    def __init__(self, path: str, device: str = "cuda"):
        self.path = path
        self.device = device
        self._rt_by_frame: Dict[int, torch.Tensor] = {}
        self._K_by_frame: Dict[int, torch.Tensor] = {}
        self._proc_wh_by_frame: Dict[int, Tuple[int, int]] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"VGGT prior file not found: {self.path}")

        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".csv":
            self._load_csv()
        elif ext == ".npz":
            self._load_npz()
        else:
            self._load_tum()

        if len(self._rt_by_frame) == 0:
            raise RuntimeError(f"No valid pose prior entries loaded from {self.path}")

    def _load_tum(self):
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out = _tum_line_to_w2c(line.split())
                if out is None:
                    continue
                frame_id, Rt = out
                self._rt_by_frame[frame_id] = Rt.to(self.device)

    def _load_csv(self):
        # Supported CSV schema A:
        # frame_id, tx, ty, tz, qx, qy, qz, qw
        # Supported CSV schema B (camera params):
        # frame_id, r11..r33, tx,ty,tz, fx,fy,cx,cy, proc_width,proc_height
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row is None:
                    continue
                try:
                    frame_id = int(row["frame_id"])
                except (ValueError, KeyError):
                    continue

                # Camera-parameter CSV (preferred when available)
                has_rotmat = all(k in row for k in ("r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"))
                has_intr = all(k in row for k in ("fx", "fy", "cx", "cy"))
                if has_rotmat:
                    try:
                        R = torch.tensor(
                            [
                                [float(row["r11"]), float(row["r12"]), float(row["r13"])],
                                [float(row["r21"]), float(row["r22"]), float(row["r23"])],
                                [float(row["r31"]), float(row["r32"]), float(row["r33"])],
                            ],
                            dtype=torch.float32,
                        )
                        t = torch.tensor(
                            [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                            dtype=torch.float32,
                        )
                    except (ValueError, KeyError):
                        continue
                    Rt = torch.eye(4, dtype=torch.float32)
                    Rt[:3, :3] = R
                    Rt[:3, 3] = t
                    self._rt_by_frame[frame_id] = Rt.to(self.device)

                    if has_intr:
                        try:
                            fx = float(row["fx"])
                            fy = float(row["fy"])
                            cx = float(row["cx"])
                            cy = float(row["cy"])
                        except ValueError:
                            fx = fy = cx = cy = None
                        if fx is not None:
                            K = torch.tensor(
                                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                                dtype=torch.float32,
                            )
                            self._K_by_frame[frame_id] = K.to(self.device)
                            try:
                                pw = int(float(row.get("proc_width", "0")))
                                ph = int(float(row.get("proc_height", "0")))
                                if pw > 0 and ph > 0:
                                    self._proc_wh_by_frame[frame_id] = (pw, ph)
                            except ValueError:
                                pass
                    continue

                # Trajectory CSV: c2w quaternion + translation
                try:
                    tx, ty, tz = float(row["tx"]), float(row["ty"]), float(row["tz"])
                    qx, qy, qz, qw = (
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                        float(row["qw"]),
                    )
                except (ValueError, KeyError):
                    continue
                R_wc = _quat_xyzw_to_rotmat(qx, qy, qz, qw)
                t_wc = torch.tensor([tx, ty, tz], dtype=torch.float32)
                R_cw = R_wc.T
                t_cw = -R_cw @ t_wc
                Rt = torch.eye(4, dtype=torch.float32)
                Rt[:3, :3] = R_cw
                Rt[:3, 3] = t_cw
                self._rt_by_frame[frame_id] = Rt.to(self.device)

    def _load_npz(self):
        # Supported NPZ keys:
        # frame_ids, w2c [N,4,4] or c2w [N,4,4], optional K [N,3,3], optional proc_hw [N,2] (w,h)
        data = np.load(self.path, allow_pickle=True)
        if "frame_ids" not in data:
            raise RuntimeError("NPZ prior missing key: frame_ids")
        frame_ids = data["frame_ids"]
        if "w2c" in data:
            w2c_all = data["w2c"]
        elif "c2w" in data:
            c2w_all = data["c2w"]
            w2c_all = []
            for T_wc in c2w_all:
                R_wc = T_wc[:3, :3]
                t_wc = T_wc[:3, 3]
                R_cw = R_wc.T
                t_cw = -R_cw @ t_wc
                T_cw = np.eye(4, dtype=np.float32)
                T_cw[:3, :3] = R_cw
                T_cw[:3, 3] = t_cw
                w2c_all.append(T_cw)
            w2c_all = np.asarray(w2c_all)
        else:
            raise RuntimeError("NPZ prior missing w2c/c2w key")

        K_all = data["K"] if "K" in data else None
        hw_all = data["proc_hw"] if "proc_hw" in data else None

        for i, fid in enumerate(frame_ids):
            frame_id = int(fid)
            Rt = torch.tensor(w2c_all[i], dtype=torch.float32, device=self.device)
            self._rt_by_frame[frame_id] = Rt

            if K_all is not None:
                K = torch.tensor(K_all[i], dtype=torch.float32, device=self.device)
                self._K_by_frame[frame_id] = K
            if hw_all is not None:
                w = int(hw_all[i][0])
                h = int(hw_all[i][1])
                if w > 0 and h > 0:
                    self._proc_wh_by_frame[frame_id] = (w, h)

    def get_rt(self, frame_id: int) -> Optional[torch.Tensor]:
        Rt = self._rt_by_frame.get(int(frame_id))
        if Rt is None:
            return None
        return Rt.clone()

    def has_intrinsics(self) -> bool:
        return len(self._K_by_frame) > 0

    def get_intrinsics(
        self,
        frame_id: int,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> Optional[dict]:
        K = self._K_by_frame.get(int(frame_id))
        if K is None:
            return None
        K = K.clone()
        fx, fy, cx, cy = float(K[0, 0].item()), float(K[1, 1].item()), float(K[0, 2].item()), float(K[1, 2].item())

        proc_wh = self._proc_wh_by_frame.get(int(frame_id), None)
        if (
            proc_wh is not None
            and target_width is not None
            and target_height is not None
            and proc_wh[0] > 0
            and proc_wh[1] > 0
        ):
            sx = float(target_width) / float(proc_wh[0])
            sy = float(target_height) / float(proc_wh[1])
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy

        focal = torch.tensor([(fx + fy) * 0.5], dtype=torch.float32, device=self.device)
        centre = torch.tensor([cx, cy], dtype=torch.float32, device=self.device)
        return {"focal": focal, "centre": centre}
