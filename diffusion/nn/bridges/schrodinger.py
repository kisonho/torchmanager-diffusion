from ._base import D, DiffusionData, E, FastSamplingDiffusionModule, Generic, LatentDiffusionModule, Module, torch


class SchrodingerBridgeModule(LatentDiffusionModule[Module, E, D], FastSamplingDiffusionModule[Module], Generic[Module, E, D]):
    """
    The Schrodinger Bridge module for I2SB.

    - Properties:
        - beta_max: The maximum diffusion coefficient in `float`.
        - ot_ode: A `bool` flag indicating whether to use deterministic OT-ODE updates.
        - betas: The symmetric bridge beta schedule in `torch.Tensor`.
        - std_fwd: The forward-process standard deviation at each time step in `torch.Tensor`.
        - std_bwd: The backward-process standard deviation at each time step in `torch.Tensor`.
        - std_sb: The Schrodinger bridge standard deviation at each time step in `torch.Tensor`.
        - mu_x0: The coefficient of the clean sample in the bridge mean in `torch.Tensor`.
        - mu_x1: The coefficient of the condition sample in the bridge mean in `torch.Tensor`.
    """

    beta_max: float
    ot_ode: bool
    betas: torch.Tensor
    std_fwd: torch.Tensor
    std_bwd: torch.Tensor
    std_sb: torch.Tensor
    mu_x0: torch.Tensor
    mu_x1: torch.Tensor

    def __init__(self, diff_model: Module, time_steps: int, *, beta_max: float = 0.3, ot_ode: bool = False, encoder: E = None, decoder: D = None) -> None:
        """
        Initialize the Schrodinger Bridge module.

        - Parameters:
            - diff_model: The diffusion model in `torch.nn.Module`.
            - time_steps: The number of time steps in `int`.
            - beta_max: The maximum diffusion coefficient in `float`.
            - ot_ode: A `bool` flag indicating whether to use deterministic OT-ODE updates.
            - encoder: The encoder model in `torch.nn.Module`.
            - decoder: The decoder model in `torch.nn.Module`.
        """
        super().__init__(diff_model, time_steps, encoder=encoder, decoder=decoder)
        self.beta_max = beta_max
        self.ot_ode = ot_ode

        # Precompute the bridge schedule once so every forward/sampling step can
        # just gather the per-time coefficients it needs.
        betas = self._build_betas(time_steps, beta_max)
        std_fwd = torch.sqrt(torch.cumsum(betas, dim=0))
        std_bwd = torch.sqrt(torch.flip(torch.cumsum(torch.flip(betas, dims=(0,)), dim=0), dims=(0,)))
        mu_x0, mu_x1, std_sb = self._compute_bridge_statistics(std_fwd, std_bwd)

        self.register_buffer("betas", betas)
        self.register_buffer("std_fwd", std_fwd)
        self.register_buffer("std_bwd", std_bwd)
        self.register_buffer("std_sb", std_sb)
        self.register_buffer("mu_x0", mu_x0)
        self.register_buffer("mu_x1", mu_x1)

    @staticmethod
    def _build_betas(time_steps: int, beta_max: float) -> torch.Tensor:
        linear_start = 1e-4
        linear_end = beta_max / time_steps
        base = torch.linspace(linear_start**0.5, linear_end**0.5, time_steps, dtype=torch.float32).pow(2)
        midpoint = (time_steps + 1) // 2
        # Mirror the first half to match the symmetric schedule used by I2SB.
        mirrored = torch.flip(base[: time_steps // 2], dims=(0,))
        return torch.cat([base[:midpoint], mirrored], dim=0)

    @staticmethod
    def _compute_bridge_statistics(std_fwd: torch.Tensor, std_bwd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Product of the forward and backward Gaussian marginals gives the
        # bridge mean coefficients and bridge variance at each time step.
        denom = std_fwd.pow(2) + std_bwd.pow(2)
        mu_x0 = std_bwd.pow(2) / denom
        mu_x1 = std_fwd.pow(2) / denom
        std_sb = torch.sqrt((std_fwd.pow(2) * std_bwd.pow(2)) / denom)
        return mu_x0, mu_x1, std_sb

    @staticmethod
    def _expand_schedule(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while len(values.shape) < len(x.shape):
            values = values.unsqueeze(-1)
        return values

    def _gather_schedule(self, schedule: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Local code uses 1..T indexing for discrete sampling, while the stored
        # schedules are 0-indexed.
        step = (t.long() - 1).clamp(min=0, max=self.time_steps - 1)
        gathered = schedule.index_select(0, step)
        return self._expand_schedule(gathered, x)

    @staticmethod
    def _gaussian_product_coef(sigma1: torch.Tensor, sigma2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Closed-form coefficients for multiplying two isotropic Gaussians.
        denom = sigma1.pow(2) + sigma2.pow(2)
        coef1 = sigma2.pow(2) / denom
        coef2 = sigma1.pow(2) / denom
        var = (sigma1.pow(2) * sigma2.pow(2)) / denom
        return coef1, coef2, var

    def _predict_x0(self, xt: torch.Tensor, t: torch.Tensor, predicted_obj: torch.Tensor) -> torch.Tensor:
        std_fwd = self._gather_schedule(self.std_fwd, t, xt)
        return xt - std_fwd * predicted_obj

    def forward_diffusion(self, data: torch.Tensor, t: torch.Tensor | None = None, /, condition: torch.Tensor | None = None) -> tuple[DiffusionData, torch.Tensor]:
        x_start = self.encode(data)
        assert condition is not None, "Condition is required for forward diffusion."
        condition = self.encode(condition)
        assert x_start.shape == condition.shape, f"X_start and condition must have the same shape, got x_start={x_start.shape} and condition={condition.shape}."

        if t is None:
            t = torch.randint(1, self.time_steps + 1, (x_start.shape[0],), device=x_start.device).long()
        else:
            t = t.to(x_start.device).long()

        mu_x0 = self._gather_schedule(self.mu_x0, t, x_start)
        mu_x1 = self._gather_schedule(self.mu_x1, t, x_start)
        std_sb = self._gather_schedule(self.std_sb, t, x_start)

        # Sample x_t from q(x_t | x_0, x_1); OT-ODE keeps only the mean path.
        xt = mu_x0 * x_start + mu_x1 * condition
        if not self.ot_ode:
            xt = xt + std_sb * torch.randn_like(x_start)

        # I2SB trains on the normalized residual that recovers x_0 from x_t.
        objective = (xt - x_start) / self._gather_schedule(self.std_fwd, t, x_start)
        return DiffusionData(xt, t, condition=condition), objective

    def sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.fast_sampling_steps is not None:
            i = len(self.fast_sampling_steps) - i
            tau = self.fast_sampling_steps[i]
            tau_minus_one = self.fast_sampling_steps[i + 1] if i < len(self.fast_sampling_steps) - 1 else 0
            t = torch.full((data.x.shape[0],), tau, device=data.x.device, dtype=torch.long)
            data = DiffusionData(data.x, t, condition=data.condition)
            return self.fast_sampling_step(data, tau, tau_minus_one, return_noise=return_noise, predicted_obj=predicted_obj)

        assert data.condition is not None, "Condition must be given for Schrodinger Bridge."
        predicted_obj = self.forward(data) if predicted_obj is None else predicted_obj
        # Convert the network prediction back into the clean endpoint x_0.
        pred_x0 = self._predict_x0(data.x, data.t, predicted_obj)

        if torch.all(data.t == 1):
            return (pred_x0, predicted_obj) if return_noise else pred_x0

        t = data.t.long()
        prev_t = t - 1
        std_n = self._gather_schedule(self.std_fwd, t, data.x)
        std_prev = self._gather_schedule(self.std_fwd, prev_t, data.x)
        std_delta = torch.sqrt(torch.clamp(std_n.pow(2) - std_prev.pow(2), min=0))
        # Use the closed-form bridge posterior p(x_{t-1} | x_t, x_0_hat).
        coef_x0, coef_xt, var = self._gaussian_product_coef(std_prev, std_delta)

        x_t_minus_one = coef_x0 * pred_x0 + coef_xt * data.x
        if not self.ot_ode:
            noise_mask = self._expand_schedule((prev_t > 1).to(data.x.dtype), data.x)
            x_t_minus_one = x_t_minus_one + noise_mask * torch.sqrt(var) * torch.randn_like(data.x)
        return (x_t_minus_one, predicted_obj) if return_noise else x_t_minus_one

    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert data.condition is not None, "Condition must be given for Schrodinger Bridge."
        predicted_obj = self.forward(data) if predicted_obj is None else predicted_obj
        pred_x0 = self._predict_x0(data.x, data.t, predicted_obj)

        if tau_minus_one == 0:
            return (pred_x0, predicted_obj) if return_noise else pred_x0

        tau_tensor = torch.full((data.x.shape[0],), tau, device=data.x.device, dtype=torch.long)
        tau_minus_one_tensor = torch.full((data.x.shape[0],), tau_minus_one, device=data.x.device, dtype=torch.long)

        std_tau = self._gather_schedule(self.std_fwd, tau_tensor, data.x)
        std_tau_minus_one = self._gather_schedule(self.std_fwd, tau_minus_one_tensor, data.x)
        std_delta = torch.sqrt(torch.clamp(std_tau.pow(2) - std_tau_minus_one.pow(2), min=0))
        # The same posterior update also works for skipped time steps.
        coef_x0, coef_xt, var = self._gaussian_product_coef(std_tau_minus_one, std_delta)

        x_tau_minus_one = coef_x0 * pred_x0 + coef_xt * data.x
        if not self.ot_ode and tau_minus_one > 1:
            x_tau_minus_one = x_tau_minus_one + torch.sqrt(var) * torch.randn_like(data.x)
        return (x_tau_minus_one, predicted_obj) if return_noise else x_tau_minus_one


__all__ = ["SchrodingerBridgeModule"]
