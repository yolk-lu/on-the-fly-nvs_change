from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import torch
import torch.nn.functional as F


@dataclass
class AnchorRetrievalCandidate:
    anchor_id: int
    score: float
    threshold: float
    keyframe_index: int


@dataclass
class RelocalizationCandidate:
    anchor_id: int
    keyframe_index: int
    frame_id: int
    score: float
    threshold: float
    triangulated_points: int
    camera_centre: list[float]
    descriptor_backend: str
    vlad_ready: bool


class DINOv2GlobalDescriptorExtractor:
    """
    DINOv2-backed global descriptor extractor with a deterministic pooled-token
    fallback. The fallback keeps tests and CPU-only smoke runs independent of
    model downloads while preserving the same descriptor/index contract.
    """

    def __init__(
        self,
        aggregator: "VLADAggregator | None" = None,
        descriptor_dim: int = 64,
        use_dinov2: bool = False,
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.descriptor_dim = int(descriptor_dim)
        self.aggregator = aggregator or VLADAggregator(num_clusters=8, descriptor_dim=self.descriptor_dim)
        self.model = None
        self.requested_dinov2 = bool(use_dinov2)
        self.last_backend = "fallback"
        if use_dinov2:
            self.model = self._load_dinov2()
        self.last_backend = "dinov2" if self.model is not None else "fallback"

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.extract_tokens(image)
        return self.descriptor_from_tokens(tokens)

    @torch.no_grad()
    def descriptor_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.aggregator.is_ready:
            descriptor = self.aggregator.encode(tokens)
        else:
            descriptor = self.mean_pool_descriptor(tokens)
        return F.normalize(descriptor.flatten(), dim=0)

    @torch.no_grad()
    def extract_tokens(self, image: torch.Tensor) -> torch.Tensor:
        image = image.detach().to(self.device).float()
        if image.ndim == 3:
            image = image[None]
        if self.model is not None:
            try:
                resized = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
                if hasattr(self.model, "forward_features"):
                    features = self.model.forward_features(resized)
                    tokens = features.get("x_norm_patchtokens", None)
                    if tokens is not None:
                        self.last_backend = "dinov2"
                        return self._project_dim(tokens[0].float())
                if hasattr(self.model, "get_intermediate_layers"):
                    features = self.model.get_intermediate_layers(
                        resized,
                        [len(getattr(self.model, "blocks", [])) - 1],
                        reshape=False,
                        return_class_token=False,
                    )
                    if len(features) > 0:
                        self.last_backend = "dinov2"
                        return self._project_dim(features[-1][0].float())
            except Exception:
                self.model = None

        self.last_backend = "fallback"
        pooled = F.adaptive_avg_pool2d(image, output_size=(8, 8))[0]
        tokens = pooled.flatten(1).T.contiguous()
        return self._project_dim(tokens)

    def mean_pool_descriptor(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.aggregator._normalize_tokens(tokens)
        if tokens.shape[0] == 0:
            return torch.zeros(self.descriptor_dim, device=self.device)
        return F.normalize(tokens.mean(dim=0), dim=0)

    def _project_dim(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] == self.descriptor_dim:
            return tokens
        if tokens.shape[-1] > self.descriptor_dim:
            return tokens[..., : self.descriptor_dim].contiguous()
        pad = self.descriptor_dim - tokens.shape[-1]
        return F.pad(tokens, (0, pad)).contiguous()

    def _load_dinov2(self):
        try:
            sys.path.append("submodules/Depth-Anything-V2")
            from depth_anything_v2.dinov2 import DINOv2

            model = DINOv2(model_name="vitl")
            model_path = "models/depth_anything_v2_vitl.pth"
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                dino_state = {
                    key[len("pretrained.") :]: value
                    for key, value in state_dict.items()
                    if key.startswith("pretrained.")
                }
                if dino_state:
                    model.load_state_dict(dino_state, strict=False)
            model.eval().to(self.device)
            return model
        except Exception:
            pass
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
            model.eval().to(self.device)
            return model
        except Exception:
            return None


class VLADAggregator:
    def __init__(self, num_clusters: int = 8, descriptor_dim: int = 64, warmup_keyframes: int = 8):
        self.num_clusters = int(num_clusters)
        self.descriptor_dim = int(descriptor_dim)
        self.warmup_keyframes = int(warmup_keyframes)
        self.centroids: torch.Tensor | None = None
        self.frozen = False
        self._warmup_tokens: list[torch.Tensor] = []

    @property
    def is_ready(self) -> bool:
        return self.centroids is not None and self.frozen

    @torch.no_grad()
    def observe(self, tokens: torch.Tensor) -> bool:
        if self.frozen:
            return False
        tokens = self._normalize_tokens(tokens).detach().cpu()
        if tokens.shape[0] == 0:
            return False
        self._warmup_tokens.append(tokens)
        if len(self._warmup_tokens) >= max(self.warmup_keyframes, 1):
            self.fit(torch.cat(self._warmup_tokens, dim=0))
            self.frozen = self.centroids is not None
            self._warmup_tokens.clear()
            return self.frozen
        return False

    @torch.no_grad()
    def fit(self, tokens: torch.Tensor) -> None:
        tokens = self._normalize_tokens(tokens)
        if tokens.shape[0] == 0:
            return
        if tokens.shape[0] >= self.num_clusters:
            indices = torch.linspace(0, tokens.shape[0] - 1, self.num_clusters, device=tokens.device).long()
            self.centroids = tokens[indices].contiguous()
        else:
            repeat = (self.num_clusters + tokens.shape[0] - 1) // tokens.shape[0]
            self.centroids = tokens.repeat(repeat, 1)[: self.num_clusters].contiguous()

    @torch.no_grad()
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._normalize_tokens(tokens)
        if self.centroids is None:
            return torch.zeros(self.num_clusters * self.descriptor_dim, device=tokens.device)
        centroids = self.centroids.to(tokens.device, tokens.dtype)
        distances = torch.cdist(tokens, centroids)
        assignment = distances.argmin(dim=1)
        residuals = []
        for cluster_id in range(self.num_clusters):
            mask = assignment == cluster_id
            if mask.any():
                residual = (tokens[mask] - centroids[cluster_id]).sum(dim=0)
            else:
                residual = torch.zeros(self.descriptor_dim, dtype=tokens.dtype, device=tokens.device)
            residuals.append(residual)
        vlad = torch.stack(residuals, dim=0)
        vlad = F.normalize(vlad, dim=1)
        return F.normalize(vlad.flatten(), dim=0)

    def _normalize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.float()
        if tokens.ndim != 2:
            tokens = tokens.reshape(-1, tokens.shape[-1])
        if tokens.shape[-1] != self.descriptor_dim:
            if tokens.shape[-1] > self.descriptor_dim:
                tokens = tokens[:, : self.descriptor_dim]
            else:
                tokens = F.pad(tokens, (0, self.descriptor_dim - tokens.shape[-1]))
        return F.normalize(tokens, dim=1)


class AdaptiveSimilarityThreshold:
    def __init__(self, min_similarity: float = 0.55, std_factor: float = 0.5, history_size: int = 256):
        self.min_similarity = float(min_similarity)
        self.std_factor = float(std_factor)
        self.history_size = int(history_size)
        self.history: list[float] = []

    def value(self) -> float:
        if len(self.history) < 4:
            return self.min_similarity
        scores = torch.tensor(self.history[-self.history_size :], dtype=torch.float32)
        return float(max(self.min_similarity, scores.mean().item() + self.std_factor * scores.std(unbiased=False).item()))

    def observe(self, scores: list[float]) -> None:
        self.history.extend(float(score) for score in scores if torch.isfinite(torch.tensor(score)))
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size :]


class AnchorDescriptorIndex:
    def __init__(
        self,
        extractor: DINOv2GlobalDescriptorExtractor | None = None,
        threshold: AdaptiveSimilarityThreshold | None = None,
        top_k: int = 8,
    ):
        self.extractor = extractor or DINOv2GlobalDescriptorExtractor(use_dinov2=False)
        self.threshold = threshold or AdaptiveSimilarityThreshold()
        self.top_k = int(top_k)
        self.records: list[dict] = []

    @torch.no_grad()
    def add_keyframe(self, keyframe, anchor_id: int) -> None:
        tokens = self.extractor.extract_tokens(keyframe.frame.image).detach().cpu()
        became_ready = self.extractor.aggregator.observe(tokens)
        descriptor = self.extractor.descriptor_from_tokens(tokens).detach().cpu()
        desc = getattr(keyframe, "desc_kpts", None)
        triangulated_points = 0
        if desc is not None and getattr(desc, "has_pt3d", None) is not None:
            triangulated_points = int(desc.has_pt3d.sum().detach().cpu().item())
        centre = []
        try:
            centre = keyframe.get_centre(approx=True).detach().cpu().tolist()
        except Exception:
            pass
        self.records.append(
            {
                "anchor_id": int(anchor_id),
                "keyframe_index": int(keyframe.index),
                "frame_id": int(keyframe.frame.frame_id),
                "tokens": tokens,
                "descriptor": descriptor,
                "triangulated_points": int(triangulated_points),
                "camera_centre": centre,
                "descriptor_backend": self.extractor.last_backend,
                "vlad_ready": bool(self.extractor.aggregator.is_ready),
            }
        )
        if became_ready:
            self._refresh_descriptors()

    def query_anchor(
        self,
        anchor_id: int,
        exclude_anchor_window: int = 2,
        top_k: int | None = None,
    ) -> list[AnchorRetrievalCandidate]:
        records = [record for record in self.records if int(record["anchor_id"]) == int(anchor_id)]
        if len(records) == 0:
            return []
        query = F.normalize(torch.stack([record["descriptor"] for record in records], dim=0).mean(dim=0), dim=0)
        scored: dict[int, tuple[float, int]] = {}
        raw_scores = []
        for record in self.records:
            dst_id = int(record["anchor_id"])
            if dst_id == int(anchor_id) or abs(dst_id - int(anchor_id)) < int(exclude_anchor_window):
                continue
            score = float(torch.dot(query, F.normalize(record["descriptor"], dim=0)).item())
            raw_scores.append(score)
            if dst_id not in scored or score > scored[dst_id][0]:
                scored[dst_id] = (score, int(record["keyframe_index"]))
        self.threshold.observe(raw_scores)
        threshold = self.threshold.value()
        candidates = [
            AnchorRetrievalCandidate(anchor_id=dst_id, score=score, threshold=threshold, keyframe_index=kf_idx)
            for dst_id, (score, kf_idx) in scored.items()
            if score >= threshold
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: (self.top_k if top_k is None else int(top_k))]

    @torch.no_grad()
    def query_frame(
        self,
        frame,
        current_anchor_id: int,
        exclude_anchor_window: int = 2,
        top_k: int | None = None,
    ) -> list[AnchorRetrievalCandidate]:
        if len(self.records) == 0:
            return []
        query = self.extractor.extract(frame.image).detach().cpu()
        query = F.normalize(query, dim=0)
        scored: dict[int, tuple[float, int]] = {}
        raw_scores = []
        for record in self.records:
            dst_id = int(record["anchor_id"])
            if abs(dst_id - int(current_anchor_id)) < int(exclude_anchor_window):
                continue
            score = float(torch.dot(query, F.normalize(record["descriptor"], dim=0)).item())
            raw_scores.append(score)
            if dst_id not in scored or score > scored[dst_id][0]:
                scored[dst_id] = (score, int(record["keyframe_index"]))
        self.threshold.observe(raw_scores)
        threshold = self.threshold.value()
        candidates = [
            AnchorRetrievalCandidate(anchor_id=dst_id, score=score, threshold=threshold, keyframe_index=kf_idx)
            for dst_id, (score, kf_idx) in scored.items()
            if score >= threshold
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: (self.top_k if top_k is None else int(top_k))]

    @torch.no_grad()
    def query_frame_for_relocalization(
        self,
        frame,
        top_k: int = 16,
        exclude_self_frame_id: int | None = None,
    ) -> list[RelocalizationCandidate]:
        if len(self.records) == 0:
            return []
        tokens = self.extractor.extract_tokens(frame.image).detach().cpu()
        query = self.extractor.descriptor_from_tokens(tokens).detach().cpu()
        query = F.normalize(query, dim=0)
        threshold = self.threshold.value()
        candidates = []
        raw_scores = []
        for record in self.records:
            if exclude_self_frame_id is not None and int(record["frame_id"]) == int(exclude_self_frame_id):
                continue
            descriptor = F.normalize(record["descriptor"], dim=0)
            if descriptor.shape != query.shape:
                continue
            score = float(torch.dot(query, descriptor).item())
            if str(record.get("descriptor_backend", "fallback")) == "fallback":
                score *= 0.8
            raw_scores.append(score)
            candidates.append(
                RelocalizationCandidate(
                    anchor_id=int(record["anchor_id"]),
                    keyframe_index=int(record["keyframe_index"]),
                    frame_id=int(record["frame_id"]),
                    score=score,
                    threshold=threshold,
                    triangulated_points=int(record.get("triangulated_points", 0)),
                    camera_centre=list(record.get("camera_centre", [])),
                    descriptor_backend=str(record.get("descriptor_backend", "fallback")),
                    vlad_ready=bool(record.get("vlad_ready", False) and self.extractor.aggregator.is_ready),
                )
            )
        self.threshold.observe(raw_scores)
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: int(top_k)]

    @torch.no_grad()
    def _refresh_descriptors(self) -> None:
        for record in self.records:
            descriptor = self.extractor.descriptor_from_tokens(record["tokens"]).detach().cpu()
            record["descriptor"] = descriptor
            record["vlad_ready"] = bool(self.extractor.aggregator.is_ready)
