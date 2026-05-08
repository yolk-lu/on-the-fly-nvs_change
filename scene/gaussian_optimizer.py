from __future__ import annotations

import torch

from scene.optimizers import BaseAdam, SparseGaussianAdam
from scene.tsdf_losses import anisotropy_regularization


class GaussianOptimizer:
    """Wrapper that preserves existing CUDA Adam optimizers and adds regularizers."""

    def __init__(self, gaussian_params: dict, lr_dict: dict | None = None, betas=(0.5, 0.99)):
        self.gaussian_params = gaussian_params
        device = gaussian_params["xyz"]["val"].device
        self.optimizer = None
        if device.type == "cuda":
            self.optimizer = SparseGaussianAdam(gaussian_params, betas=betas, lr_dict={} if lr_dict is None else lr_dict)

    def zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step(self, visibility: torch.Tensor, n_gaussians: int) -> None:
        if self.optimizer is None:
            raise RuntimeError("SparseGaussianAdam requires CUDA Gaussian parameters")
        self.optimizer.step(visibility, n_gaussians)

    def add_and_prune(self, extension_tensors: dict[str, torch.Tensor], valid_mask: torch.Tensor) -> None:
        if self.optimizer is None:
            raise RuntimeError("SparseGaussianAdam requires CUDA Gaussian parameters")
        self.optimizer.add_and_prune(extension_tensors, valid_mask)

    def anisotropy_loss(self, max_ratio: float = 8.0) -> torch.Tensor:
        return anisotropy_regularization(self.gaussian_params["scaling"]["val"], max_ratio=max_ratio)


class PoseDepthOptimizer:
    """Thin alias around BaseAdam for camera/depth/exposure parameter groups."""

    def __init__(self, params: dict, betas=(0.8, 0.99)):
        self.optimizer = BaseAdam(params, betas=betas)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()
