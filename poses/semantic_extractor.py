#
# Semantic feature extraction using DINOv2 backbone for improved matching
# under large viewpoint changes.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append("submodules/Depth-Anything-V2")
from depth_anything_v2.dinov2 import DINOv2

from poses.feature_detector import InterpolateSparse2d


class SemanticExtractor:
    """
    Extracts semantic features at keypoint locations using DINOv2 backbone.
    Uses PCA to reduce dimensionality from 1024D to sem_dim (default 128D).
    """

    @torch.no_grad()
    def __init__(self, width: int, height: int, sem_dim: int = 128):
        self.width = width
        self.height = height
        self.sem_dim = sem_dim
        self.input_size = 518  # Same as Depth-Anything-V2

        # Load DINOv2 vitl backbone (same weights as Depth-Anything-V2)
        model = DINOv2(model_name="vitl")
        model_path = "models/depth_anything_v2_vitl.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            # Filter to only load the pretrained (DINOv2) weights
            prefix = "pretrained."
            dino_state = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if dino_state:
                model.load_state_dict(dino_state, strict=False)

        self.model = model.cuda().half().eval()
        self.embed_dim = model.embed_dim  # 1024 for vitl
        # Use the last intermediate layer for richest semantic features
        self.intermediate_layer_idx = [23]  # Last layer for vitl

        self.interpolator = InterpolateSparse2d("bilinear")

        # PCA projection matrix: initialized on first call
        self.pca_projection = None  # [embed_dim, sem_dim]

        # ImageNet normalization (same as Depth-Anything-V2)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1).half()
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1).half()

    @torch.no_grad()
    def _init_pca(self, features: torch.Tensor):
        """Initialize PCA projection from first batch of features. features: [N, embed_dim]"""
        # Center the features
        mean = features.mean(dim=0)
        centered = features - mean
        # SVD for PCA
        U, S, Vh = torch.linalg.svd(centered.float(), full_matrices=False)
        # Take top sem_dim components
        self.pca_mean = mean.half()
        self.pca_projection = Vh[:self.sem_dim].T.half()  # [embed_dim, sem_dim]

    @torch.no_grad()
    def extract(self, image: torch.Tensor, kpts: torch.Tensor, max_kpts_per_batch: int = 2048) -> torch.Tensor:
        """
        Extract semantic features at keypoint locations.

        Args:
            image: [3, H, W] input image tensor (float, 0-1 range)
            kpts: [N, 2] keypoint positions in pixel coordinates
            max_kpts_per_batch: batch size for keypoint interpolation

        Returns:
            sem_feats: [N, sem_dim] L2-normalized semantic features
        """
        # Resize to DINOv2 input size
        img = F.interpolate(
            image[None].half(),
            (self.input_size, self.input_size),
            mode="bilinear",
            align_corners=True,
        )
        # Normalize with ImageNet stats
        img = (img - self.mean) / self.std

        # Forward through DINOv2 backbone
        patch_h = self.input_size // 14
        patch_w = self.input_size // 14

        features = self.model.get_intermediate_layers(
            img, self.intermediate_layer_idx, reshape=True, return_class_token=False
        )
        # features is a tuple of tensors, each [B, C, patch_h, patch_w]
        feat_map = features[-1]  # [1, embed_dim, patch_h, patch_w]

        # Interpolate at keypoint locations (in batches to save memory)
        N = kpts.shape[0]
        sem_feats_list = []

        for start in range(0, N, max_kpts_per_batch):
            end = min(start + max_kpts_per_batch, N)
            batch_kpts = kpts[start:end]  # [batch, 2]

            # Scale keypoints from image coords to feature map coords
            scale_x = (patch_w * 14) / self.width
            scale_y = (patch_h * 14) / self.height
            scaled_kpts = batch_kpts.clone()
            scaled_kpts[:, 0] *= scale_x
            scaled_kpts[:, 1] *= scale_y

            # Interpolate features at keypoint positions
            batch_feats = self.interpolator(
                feat_map, scaled_kpts[None], patch_h * 14, patch_w * 14
            )[0]  # [batch, embed_dim]
            sem_feats_list.append(batch_feats)

        sem_feats = torch.cat(sem_feats_list, dim=0)  # [N, embed_dim]

        # PCA dimensionality reduction
        if self.pca_projection is None:
            self._init_pca(sem_feats)

        sem_feats = (sem_feats - self.pca_mean) @ self.pca_projection  # [N, sem_dim]

        # L2 normalize
        sem_feats = F.normalize(sem_feats, dim=-1)

        return sem_feats
