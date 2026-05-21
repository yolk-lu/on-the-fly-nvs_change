#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
import sys

from poses.ransac import EstimatorType, RANSACEstimator


class Matches:
    """
    A class to store matched keypoints and their indices between two sets of keypoints.
    """
    def __init__(self, kpts, kpts_other, idx, idx_other):
        self.kpts = kpts
        self.kpts_other = kpts_other
        self.idx = idx
        self.idx_other = idx_other


# Adapted from https://github.com/verlab/accelerated_features
def match(feats1, feats2, min_cossim=0.82,
          sem_feats1=None, sem_feats2=None, sem_weight=0.0):
    # torch.cuda.empty_cache()
    xfeat_cossim = feats1 @ feats2.t()

    # Use fused score for mutual nearest neighbor assignment,
    # but keep XFeat-only score for threshold filtering (better calibrated)
    if sem_feats1 is not None and sem_feats2 is not None and sem_weight > 0:
        sem_cossim = sem_feats1 @ sem_feats2.t()
        cossim = (1 - sem_weight) * xfeat_cossim + sem_weight * sem_cossim
    else:
        cossim = xfeat_cossim

    bestcossim_fused, match12 = cossim.max(dim=1)
    _, match21 = cossim.max(dim=0)

    idx0 = torch.arange(match12.shape[0], device=match12.device)
    mask = match21[match12] == idx0

    # Apply threshold on XFeat score only (well-calibrated, not affected by PCA quality)
    if min_cossim > 0:
        bestcossim_xfeat = xfeat_cossim[idx0, match12]
        mask *= bestcossim_xfeat > min_cossim

    return idx0, match12, mask



class Matcher:
    @torch.no_grad()
    def __init__(
        self,
        fundmat_samples: int,
        max_error: float,
        sem_weight: float = 0.0,
        matcher_backend: str = "mnn",
        feature_backend: str = "xfeat",
        lightglue_filter_threshold: float = 0.1,
        lightglue_depth_confidence: float = 0.95,
        lightglue_width_confidence: float = 0.99,
    ):
        """
        Initialize the Matcher.
        Args:
            fundmat_samples (int): Number of RANSAC etimations when estimating inliers with fundamental matrix estimation.
            max_error (float): Maximum error for RANSAC inlier threshold.
            sem_weight (float): Weight of semantic similarity in matching score (0-1).
            matcher_backend (str): Matching backend: "mnn" or "lightglue".
            feature_backend (str): Feature extractor backend used with matcher.
        """
        self.max_error = max_error
        self.sem_weight = sem_weight
        self.matcher_backend = matcher_backend.lower()
        self.feature_backend = feature_backend.lower()
        self.fundmat_estimator = RANSACEstimator(
            fundmat_samples, max_error, EstimatorType.FUNDAMENTAL_8PTS
        )
        self.lightglue_matcher = None

        if self.matcher_backend == "lightglue":
            supported_features = {"superpoint", "disk", "sift", "aliked"}
            if self.feature_backend not in supported_features:
                raise ValueError(
                    f"LightGlue matcher requires feature_backend in {sorted(supported_features)}, "
                    f"got '{self.feature_backend}'."
                )
            lg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submodules", "LightGlue"))
            if lg_root not in sys.path:
                sys.path.insert(0, lg_root)
            from lightglue import LightGlue

            self.lightglue_matcher = LightGlue(
                features=self.feature_backend,
                filter_threshold=lightglue_filter_threshold,
                depth_confidence=lightglue_depth_confidence,
                width_confidence=lightglue_width_confidence,
            ).eval().cuda()
        elif self.matcher_backend != "mnn":
            raise ValueError(
                f"Unsupported matcher backend '{self.matcher_backend}'. Use one of: mnn, lightglue."
            )

    def _resolve_image_size(self, desc_kpts: 'DescribedKeypoints'):
        image_size = desc_kpts.meta.get("image_size") if hasattr(desc_kpts, "meta") else None
        if isinstance(image_size, torch.Tensor):
            return image_size.float()
        if len(desc_kpts.kpts) == 0:
            return torch.tensor([1.0, 1.0], device=desc_kpts.kpts.device)
        min_xy = desc_kpts.kpts.min(dim=0).values
        max_xy = desc_kpts.kpts.max(dim=0).values
        # image_size follows [width, height]
        wh = (max_xy - min_xy + 1.0).clamp_min(1.0)
        return wh.float()

    def _lightglue_match(self, desc_kpts: 'DescribedKeypoints', desc_kpts_other: 'DescribedKeypoints'):
        out_device = torch.device("cuda" if torch.cuda.is_available() else desc_kpts.kpts.device)
        if len(desc_kpts.kpts) == 0 or len(desc_kpts_other.kpts) == 0:
            empty = torch.empty(0, dtype=torch.long, device=out_device)
            return empty, empty, torch.empty(0, dtype=torch.bool, device=out_device)

        image0 = {
            "keypoints": desc_kpts.kpts[None].float().cuda(),
            "descriptors": desc_kpts.feats[None].float().cuda(),
            "image_size": self._resolve_image_size(desc_kpts)[None].float().cuda(),
        }
        image1 = {
            "keypoints": desc_kpts_other.kpts[None].float().cuda(),
            "descriptors": desc_kpts_other.feats[None].float().cuda(),
            "image_size": self._resolve_image_size(desc_kpts_other)[None].float().cuda(),
        }
        for key in ("scales", "oris"):
            value0 = desc_kpts.meta.get(key) if hasattr(desc_kpts, "meta") else None
            value1 = desc_kpts_other.meta.get(key) if hasattr(desc_kpts_other, "meta") else None
            if isinstance(value0, torch.Tensor) and isinstance(value1, torch.Tensor):
                image0[key] = value0[None].float().cuda()
                image1[key] = value1[None].float().cuda()

        matched = self.lightglue_matcher({"image0": image0, "image1": image1})
        pair_idx = matched["matches"][0]
        if pair_idx.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=out_device)
            return empty, empty, torch.empty(0, dtype=torch.bool, device=out_device)

        idx = pair_idx[:, 0].long()
        idx_other = pair_idx[:, 1].long()
        mask = torch.ones_like(idx, dtype=torch.bool)
        return idx, idx_other, mask

    def _mnn_match(self, desc_kpts: 'DescribedKeypoints', desc_kpts_other: 'DescribedKeypoints'):
        return match(
            desc_kpts.feats.cuda(), desc_kpts_other.feats.cuda(),
            sem_feats1=desc_kpts.sem_feats.cuda() if desc_kpts.sem_feats is not None else None,
            sem_feats2=desc_kpts_other.sem_feats.cuda() if desc_kpts_other.sem_feats is not None else None,
            sem_weight=self.sem_weight,
        )

    def evaluate_match(
        self, desc_kpts: 'DescribedKeypoints', desc_kpts_other: 'DescribedKeypoints'
    ):
        """
        Get the number of matches between two sets of described keypoints.
        """
        if self.matcher_backend == "lightglue":
            idx, _, _ = self._lightglue_match(desc_kpts, desc_kpts_other)
            device = "cuda" if torch.cuda.is_available() else idx.device
            return torch.tensor(len(idx), device=device)
        _, _, mask = self._mnn_match(desc_kpts, desc_kpts_other)
        return mask.sum()

    @torch.no_grad()
    def __call__(
        self,
        desc_kpts: 'DescribedKeypoints',
        desc_kpts_other: 'DescribedKeypoints',
        remove_outliers: bool = False,
        update_kpts_flag: str = "",
        kID: int = -1,
        kID_other: int = -1,
    ):
        """
        Matches keypoints between two sets of described keypoints, with optional outlier removal based on the fundamental RANSAC estimation.
        Args:
            desc_kpts (DescribedKeypoints): Keypoints and descriptors of the first image.
            desc_kpts_other (DescribedKeypoints): Keypoints and descriptors of the second image.
            remove_outliers (bool): Whether to remove outliers using the fundamental matrix.
            update_kpts_flag (str): If "all", updates all matches; if "inliers", updates only inliers.
            kID (int): ID of the first set of keypoints, used for updating matches.
            kID_other (int): ID of the second set of keypoints, used for updating matches.
        Returns:
            Matches: A Matches object containing the matched keypoints and their indices.
        """
        if self.matcher_backend == "lightglue":
            idx, idx_other, mask = self._lightglue_match(desc_kpts, desc_kpts_other)
        else:
            idx, idx_other, mask = self._mnn_match(desc_kpts, desc_kpts_other)
        idx = idx[mask]
        idx_other = idx_other[mask]
        kpts = desc_kpts.kpts[idx]
        kpts_other = desc_kpts_other.kpts[idx_other]
        idx_all = idx
        idx_other_all = idx_other
        kpts_all = kpts
        kpts_other_all = kpts_other

        if remove_outliers and len(kpts) >= self.fundmat_estimator.m:
            F, mask = self.fundmat_estimator(kpts, kpts_other)
            idx = idx[mask]
            idx_other = idx_other[mask]
            kpts = kpts[mask]
            kpts_other = kpts_other[mask]

        if update_kpts_flag == "all":
            assert kID >= 0 and kID_other >= 0
            desc_kpts.update_matches(
                kID_other, Matches(kpts_all, kpts_other_all, idx_all, idx_other_all)
            )
            desc_kpts_other.update_matches(
                kID, Matches(kpts_other_all, kpts_all, idx_other_all, idx_all)
            )
        elif update_kpts_flag == "inliers":
            assert kID >= 0 and kID_other >= 0
            desc_kpts.update_matches(
                kID_other, Matches(kpts, kpts_other, idx, idx_other)
            )
            desc_kpts_other.update_matches(
                kID, Matches(kpts_other, kpts, idx_other, idx)
            )

        return Matches(kpts, kpts_other, idx, idx_other)
