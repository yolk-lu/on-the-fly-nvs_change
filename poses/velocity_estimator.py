import torch


class LSFVelocityEstimator:
    """
    Least-squares fit (LSF) velocity estimator.

    Uses an Nth-order polynomial over an M-sample window (M > N+1), and evaluates
    derivative at the newest sample x=M:
      h = (A^+)^T qdot
      v = sum_i h_i * c_i / dt
    where c_i are camera centers ordered oldest -> newest.
    """

    def __init__(
        self,
        order: int = 2,
        window: int = 8,
        dt: float = 1.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        if window <= order + 1:
            raise ValueError(f"LSF requires M > N+1, got N={order}, M={window}")
        self.order = int(order)
        self.window = int(window)
        self.dt = float(dt)
        self.device = device
        self.dtype = dtype
        self._weight_cache = {}
        self.get_weights(
            order=self.order,
            window=self.window,
            device=self.device,
            dtype=self.dtype,
        )

    @staticmethod
    def _cache_key(order: int, window: int, device, dtype: torch.dtype):
        return (int(order), int(window), str(device), str(dtype))

    @staticmethod
    def build_weights(
        order: int,
        window: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Build derivative FIR weights:
          A[i,j] = (i+1)^j, i=0..M-1, j=0..N
          qdot[j] = j * M^(j-1), j>=1
          h = (A^+)^T qdot
        """
        if window <= order + 1:
            raise ValueError(f"LSF requires M > N+1, got N={order}, M={window}")
        x = torch.arange(1, window + 1, device=device, dtype=dtype)
        powers = [torch.ones_like(x)]
        for j in range(1, order + 1):
            powers.append(x ** j)
        A = torch.stack(powers, dim=1)  # [M, N+1]
        A_plus = torch.linalg.pinv(A)   # [N+1, M]

        qdot = torch.zeros(order + 1, device=device, dtype=dtype)
        M = float(window)
        for j in range(1, order + 1):
            qdot[j] = j * (M ** (j - 1))

        h = A_plus.transpose(0, 1) @ qdot  # [M]
        return h

    def get_weights(
        self,
        order: int,
        window: int,
        device=None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        key = self._cache_key(order, window, device, dtype)
        if key not in self._weight_cache:
            self._weight_cache[key] = self.build_weights(
                int(order), int(window), device=device, dtype=dtype
            )
        return self._weight_cache[key]

    @staticmethod
    def velocity_from_centers(
        centers: torch.Tensor,
        weights: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        centers: [M,3], ordered oldest -> newest.
        weights: [M], matching centers length.
        """
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError(f"centers should be [M,3], got {tuple(centers.shape)}")
        if centers.shape[0] != weights.shape[0]:
            raise ValueError(
                f"centers/weights length mismatch: {centers.shape[0]} vs {weights.shape[0]}"
            )
        dt_safe = max(float(dt), 1e-8)
        return (weights[:, None] * centers).sum(dim=0) / dt_safe

    def estimate(self, centers: torch.Tensor, dt: float = None) -> torch.Tensor:
        """
        Estimate velocity using available centers.
        If fewer than configured window points are provided, adapt order/window safely.
        """
        if centers.shape[0] < 3:
            return torch.zeros(3, device=centers.device, dtype=centers.dtype)
        M = int(centers.shape[0])
        N = min(self.order, max(1, M - 2))
        if M <= N + 1:
            return torch.zeros(3, device=centers.device, dtype=centers.dtype)
        weights = self.get_weights(
            order=N, window=M, device=centers.device, dtype=centers.dtype
        )
        if dt is None:
            dt = self.dt
        return self.velocity_from_centers(centers, weights, dt=dt)

    def compare_with_current(
        self,
        centers: torch.Tensor,
        c_last: torch.Tensor,
        c_candidate: torch.Tensor,
        dt: float = None,
    ) -> dict:
        """
        Compare LSF-estimated velocity and current candidate velocity.
        """
        if dt is None:
            dt = self.dt
        dt_safe = max(float(dt), 1e-8)

        v_lsf = self.estimate(centers, dt=dt_safe)
        v_curr = (c_candidate - c_last) / dt_safe

        v_lsf_norm = torch.linalg.norm(v_lsf).item()
        v_curr_norm = torch.linalg.norm(v_curr).item()
        jump_ratio = torch.linalg.norm(v_curr - v_lsf).item() / max(v_lsf_norm, 1e-8)

        denom = max(v_curr_norm * v_lsf_norm, 1e-8)
        cosang = torch.clamp(torch.dot(v_curr, v_lsf) / denom, -1.0, 1.0)
        angle_deg = float(torch.rad2deg(torch.arccos(cosang)).item())

        return {
            "velocity_lsf_norm": float(v_lsf_norm),
            "velocity_curr_norm": float(v_curr_norm),
            "velocity_jump_ratio": float(jump_ratio),
            "velocity_angle_deg": float(angle_deg),
        }
