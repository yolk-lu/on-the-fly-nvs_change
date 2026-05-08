from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RPCCompensationResult:
    keypoints: torch.Tensor
    residual: torch.Tensor
    used_rpc: bool


class RPCCompensator:
    """RPC correction interface with identity fallback when metadata is absent."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    @torch.no_grad()
    def compensate(self, keypoints: torch.Tensor, frame_info: dict, camera_prior: dict | None = None) -> RPCCompensationResult:
        del camera_prior
        if not self.enabled or "rpc" not in frame_info:
            return RPCCompensationResult(keypoints=keypoints, residual=torch.zeros_like(keypoints), used_rpc=False)
        rpc = frame_info["rpc"]
        offset = torch.as_tensor(rpc.get("pixel_offset", [0.0, 0.0]), device=keypoints.device, dtype=keypoints.dtype)
        compensated = keypoints + offset[None]
        return RPCCompensationResult(compensated, compensated - keypoints, True)
