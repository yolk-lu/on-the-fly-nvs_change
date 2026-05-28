import math
from typing import Dict, Literal, Optional, Tuple

import torch


# Model settings used by GRIC for common two-view geometry models.
# d: structure dimension, k: model parameter count
MODEL_CONFIG = {
    "essential": {"d": 3, "k": 5},
    "homography": {"d": 2, "k": 8},
}


def _to_1d_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0:
        return values.reshape(1)
    return values.reshape(-1)


class GRIC:
    """
    GRIC scorer and model selector.

    Keeps shared hyperparameters (sigma, r, margin, model config) in one place.
    """

    def __init__(
        self,
        sigma: float,
        r: int = 4,
        margin: float = 0.0,
        model_config: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = float(sigma)
        self.r = int(r)
        self.margin = float(margin)
        self.model_config = MODEL_CONFIG if model_config is None else model_config
        self.last_result: Optional[dict] = None

    def truncated_rho(self, errors_sq: torch.Tensor, *, d: int) -> torch.Tensor:
        """
        rho(e_i^2) = min( e_i^2 / (2 * (r - d) * sigma^2), 1 )
        """
        if self.r <= d:
            raise ValueError(f"GRIC requires r > d, got r={self.r}, d={d}")
        errors_sq = _to_1d_tensor(errors_sq).to(torch.float32)
        denom = 2.0 * float(self.r - d) * self.sigma * self.sigma
        return torch.clamp(errors_sq / denom, max=1.0)

    def compute(self, errors_sq: torch.Tensor, *, d: int, k: int) -> torch.Tensor:
        """
        GRIC = sum rho(e_i^2) + log(4) * d * n + log(4n) * k
        """
        errors_sq = _to_1d_tensor(errors_sq)
        n = int(errors_sq.numel())
        if n <= 0:
            return torch.tensor(float("inf"), device=errors_sq.device)

        rho_sum = self.truncated_rho(errors_sq, d=d).sum()
        complexity = math.log(4.0) * d * n + math.log(4.0 * n) * k
        return rho_sum + torch.tensor(
            complexity, device=errors_sq.device, dtype=rho_sum.dtype
        )

    def compute_for_model(
        self,
        errors_sq: torch.Tensor,
        model: Literal["essential", "homography"],
    ) -> torch.Tensor:
        if model not in self.model_config:
            raise ValueError(
                f"Unknown model '{model}'. Expected one of {list(self.model_config.keys())}"
            )
        cfg = self.model_config[model]
        return self.compute(errors_sq, d=cfg["d"], k=cfg["k"])

    def select(
        self,
        errors_sq_essential: torch.Tensor,
        errors_sq_homography: torch.Tensor,
    ) -> Tuple[str, float, float]:
        """
        Selection rule:
          choose homography if gric_h + margin < gric_e, else essential
        Returns:
          selected_model, gric_essential, gric_homography
        """
        gric_e = self.compute_for_model(errors_sq_essential, "essential")
        gric_h = self.compute_for_model(errors_sq_homography, "homography")
        selected = "homography" if (gric_h + self.margin) < gric_e else "essential"
        self.last_result = {
            "selected_model": selected,
            "gric_essential": float(gric_e.item()),
            "gric_homography": float(gric_h.item()),
        }
        return selected, float(gric_e.item()), float(gric_h.item())

    def __call__(
        self,
        errors_sq_essential: torch.Tensor,
        errors_sq_homography: torch.Tensor,
    ) -> Tuple[str, float, float]:
        return self.select(errors_sq_essential, errors_sq_homography)


# Backward-compatible functional wrappers
def truncated_rho(
    errors_sq: torch.Tensor,
    *,
    r: int = 4,
    d: int,
    sigma: float,
) -> torch.Tensor:
    return GRIC(sigma=sigma, r=r).truncated_rho(errors_sq, d=d)


def compute_gric(
    errors_sq: torch.Tensor,
    *,
    d: int,
    k: int,
    sigma: float,
    r: int = 4,
) -> torch.Tensor:
    return GRIC(sigma=sigma, r=r).compute(errors_sq, d=d, k=k)


def compute_gric_for_model(
    errors_sq: torch.Tensor,
    model: Literal["essential", "homography"],
    *,
    sigma: float,
    r: int = 4,
) -> torch.Tensor:
    return GRIC(sigma=sigma, r=r).compute_for_model(errors_sq, model)


def select_model_by_gric(
    errors_sq_essential: torch.Tensor,
    errors_sq_homography: torch.Tensor,
    *,
    sigma: float,
    r: int = 4,
    margin: float = 0.0,
) -> Tuple[str, float, float]:
    return GRIC(sigma=sigma, r=r, margin=margin).select(
        errors_sq_essential, errors_sq_homography
    )
