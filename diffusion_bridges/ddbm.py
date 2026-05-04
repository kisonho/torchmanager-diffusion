import torch
from diffusion.data import DiffusionData
from diffusion.nn import FastSamplingDiffusionModule, LatentDiffusionModule
from diffusion.sde import SDEType
from typing import TypeVar

Module = TypeVar('Module', bound=torch.nn.Module)
E = TypeVar('E', bound=torch.nn.Module | None)
D = TypeVar('D', bound=torch.nn.Module | None)


class DDBMModule(LatentDiffusionModule[Module, E, D], FastSamplingDiffusionModule[Module]):
    """
    The Denoising Diffusion Bridge Model module.

    This implementation follows the bridge-scaled denoising parameterization used
    by the official DDBM reference implementation, adapted to this package's
    discrete `forward_diffusion` / `sampling_step` API.
    """

    sigma_data: float
    sigma_min: float
    sigma_max: float
    rho: float
    beta_d: float
    beta_min: float
    cov_xy: float
    guidance: float
    churn_step_ratio: float
    pred_mode: SDEType | None
    sigma_schedule: torch.Tensor

    def __init__(self, diff_model: Module, time_steps: int, *, sigma_data: float = 0.5, sigma_min: float = 0.002, sigma_max: float = 1.0, rho: float = 7.0, beta_d: float = 2.0, beta_min: float = 0.1, cov_xy: float = 0.0, guidance: float = 1.0, churn_step_ratio: float = 0.0, pred_mode: SDEType | None = SDEType.VP, encoder: E = None, decoder: D = None) -> None:
        """
        Initialize the DDBM module.

        - Parameters:
            - diff_model: The diffusion model in `torch.nn.Module`.
            - time_steps: The number of time steps in `int`.
            - sigma_data: The data standard deviation in `float`.
            - sigma_min: The minimum bridge time in `float`.
            - sigma_max: The maximum bridge time in `float`.
            - rho: The Karras schedule exponent in `float`.
            - beta_d: The VP beta coefficient in `float`.
            - beta_min: The VP minimum beta in `float`.
            - cov_xy: The covariance term between source and target endpoints in `float`.
            - guidance: The bridge guidance coefficient in `float`.
            - churn_step_ratio: The stochastic churn ratio in `float`.
            - pred_mode: The bridge schedule mode in `diffusion.sde.SDEType`.
            - encoder: The encoder model in `torch.nn.Module`.
            - decoder: The decoder model in `torch.nn.Module`.
        """
        super().__init__(diff_model, time_steps, encoder=encoder, decoder=decoder)
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.cov_xy = cov_xy
        self.guidance = guidance
        self.churn_step_ratio = churn_step_ratio
        self.pred_mode = pred_mode

        sigma_schedule = self._build_sigma_schedule(time_steps, sigma_min, sigma_max, rho)
        self.register_buffer("sigma_schedule", sigma_schedule)

    @staticmethod
    def _expand(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while len(values.shape) < len(x.shape):
            values = values.unsqueeze(-1)
        return values

    @staticmethod
    def _append_zero(values: torch.Tensor) -> torch.Tensor:
        return torch.cat([values, torch.zeros(1, dtype=values.dtype, device=values.device)])

    @staticmethod
    def _build_sigma_schedule(time_steps: int, sigma_min: float, sigma_max: float, rho: float) -> torch.Tensor:
        ramp = torch.linspace(0, 1, time_steps, dtype=torch.float32)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return DDBMModule._append_zero(sigmas)

    @staticmethod
    def _vp_logsnr(t: torch.Tensor, beta_d: float, beta_min: float) -> torch.Tensor:
        return -torch.log(torch.expm1(0.5 * beta_d * t.pow(2) + beta_min * t))

    @staticmethod
    def _vp_logs(t: torch.Tensor, beta_d: float, beta_min: float) -> torch.Tensor:
        return -0.25 * beta_d * t.pow(2) - 0.5 * beta_min * t

    def _gather_sigma(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Map discrete training/sampling steps onto the continuous DDBM sigma schedule.
        indices = (self.time_steps - t.to(self.sigma_schedule.device).long()).clamp(min=0, max=self.time_steps)
        gathered = self.sigma_schedule.index_select(0, indices).to(x.device)
        return self._expand(gathered, x)

    def _get_bridge_scalings(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_data_end = self.sigma_data
        c = 1.0

        match self.pred_mode:
            case SDEType.VE:
                sigma_sq = sigma.pow(2)
                sigma_max_sq = self.sigma_max**2
                a = sigma_sq.pow(2) / (sigma_max_sq**2) * sigma_data_end**2
                b = (1 - sigma_sq / sigma_max_sq).pow(2) * self.sigma_data**2
                cross = 2 * sigma_sq / sigma_max_sq * (1 - sigma_sq / sigma_max_sq) * self.cov_xy
                bridge = c**2 * sigma_sq * (1 - sigma_sq / sigma_max_sq)
                total = a + b + cross + bridge
                c_in = total.rsqrt()
                c_skip = ((1 - sigma_sq / sigma_max_sq) * self.sigma_data**2 + sigma_sq / sigma_max_sq * self.cov_xy) / total
                c_out = torch.sqrt((sigma_sq / sigma_max_sq).pow(2) * (sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * c**2 * sigma_sq * (1 - sigma_sq / sigma_max_sq)) * c_in
                return c_skip, c_out, c_in
            case SDEType.VP:
                logsnr_t = self._vp_logsnr(sigma, self.beta_d, self.beta_min)
                logsnr_t_max = self._vp_logsnr(torch.ones_like(sigma), self.beta_d, self.beta_min)
                logs_t = self._vp_logs(sigma, self.beta_d, self.beta_min)
                logs_t_max = self._vp_logs(torch.ones_like(sigma), self.beta_d, self.beta_min)

                a_t = torch.exp(logsnr_t_max - logsnr_t + logs_t - logs_t_max)
                b_t = -torch.expm1(logsnr_t_max - logsnr_t) * torch.exp(logs_t)
                c_t = -torch.expm1(logsnr_t_max - logsnr_t) * torch.exp(2 * logs_t - logsnr_t)
                total = a_t.pow(2) * sigma_data_end**2 + b_t.pow(2) * self.sigma_data**2 + 2 * a_t * b_t * self.cov_xy + c**2 * c_t
                c_in = total.rsqrt()
                c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy) / total
                c_out = torch.sqrt(a_t.pow(2) * (sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * c**2 * c_t) * c_in
                return c_skip, c_out, c_in
            case _:
                ones = torch.ones_like(sigma)
                zeros = torch.zeros_like(sigma)
                return zeros, ones, ones
        raise NotImplementedError(f"Unsupported DDBM pred_mode: {self.pred_mode}")

    def _run_model(self, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        # DDBM conditions the backbone on log-sigma rather than raw integer steps.
        condition = condition.to(x.device) if condition is not None else None
        sigma = sigma.to(device=x.device, dtype=x.dtype)
        rescaled_t = 250 * torch.log(sigma.reshape(x.shape[0]) + 1e-44)
        x_in = self._get_bridge_scalings(sigma)[2] * x
        data = DiffusionData(x_in, rescaled_t, condition=condition)
        if condition is not None:
            return self.model(*data)
        return self.model(data.x, data.t)

    def _denoise(self, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert the network output back into the clean-sample estimate with bridge scalings.
        c_skip, c_out, _ = self._get_bridge_scalings(sigma)
        model_output = self._run_model(x, sigma, condition=condition)
        denoised = c_out * model_output + c_skip * x
        return model_output, denoised

    def forward(self, data: DiffusionData) -> torch.Tensor:
        sigma = self._gather_sigma(data.t, data.x)
        _, denoised = self._denoise(data.x, sigma, condition=data.condition if isinstance(data.condition, torch.Tensor) else None)
        return denoised

    def _bridge_sample(self, x_start: torch.Tensor, x_end: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.pred_mode is SDEType.VE:
            # VE mode uses the closed-form bridge between paired endpoints.
            sigma_sq = sigma.pow(2)
            sigma_max_sq = self.sigma_max**2
            std_t = sigma * torch.sqrt(torch.clamp(1 - sigma_sq / sigma_max_sq, min=0))
            mu_t = sigma_sq / sigma_max_sq * x_end + (1 - sigma_sq / sigma_max_sq) * x_start
            return mu_t + std_t * noise
        elif self.pred_mode is SDEType.VP:
            # VP mode follows the time-dependent coefficients from the reference DDBM implementation.
            sigma_flat = sigma.reshape(x_start.shape[0])
            logsnr_t = self._vp_logsnr(sigma_flat, self.beta_d, self.beta_min)
            logsnr_t_max = self._vp_logsnr(torch.full_like(sigma_flat, self.sigma_max), self.beta_d, self.beta_min)
            logs_t = self._vp_logs(sigma_flat, self.beta_d, self.beta_min)
            logs_t_max = self._vp_logs(torch.full_like(sigma_flat, self.sigma_max), self.beta_d, self.beta_min)
            a_t = torch.exp(logsnr_t_max - logsnr_t + logs_t - logs_t_max)
            b_t = -torch.expm1(logsnr_t_max - logsnr_t) * torch.exp(logs_t)
            std_t = torch.sqrt(-torch.expm1(logsnr_t_max - logsnr_t)) * torch.exp(logs_t - 0.5 * logsnr_t)
            a_t = self._expand(a_t, x_start)
            b_t = self._expand(b_t, x_start)
            std_t = self._expand(std_t, x_start)
            return a_t * x_end + b_t * x_start + std_t * noise
        raise NotImplementedError(f"Unsupported DDBM pred_mode: {self.pred_mode}")

    def forward_diffusion(self, data: torch.Tensor, t: torch.Tensor | None = None, /, condition: torch.Tensor | None = None) -> tuple[DiffusionData, torch.Tensor]:
        x_start = self.encode(data)
        assert condition is not None, "Condition is required for forward diffusion."
        x_end = self.encode(condition.to(x_start.device)).to(x_start.device)
        assert x_start.shape == x_end.shape, f"X_start and condition must have the same shape, got x_start={x_start.shape} and condition={x_end.shape}."
        if t is None:
            t = torch.randint(1, self.time_steps + 1, (x_start.shape[0],), device=x_start.device).long()
        else:
            t = t.to(x_start.device).long()

        # Training supervises the denoised x0 reconstruction from a bridged x_t sample.
        sigma = self._gather_sigma(t, x_start)
        noise = torch.randn_like(x_start)
        x_t = self._bridge_sample(x_start, x_end, sigma, noise)
        return DiffusionData(x_t, t, condition=x_end), x_start

    def _ve_derivative(self, x: torch.Tensor, denoised: torch.Tensor, x_end: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        grad_pxtlx0 = (denoised - x) / torch.clamp(sigma.pow(2), min=1e-12)
        denom = torch.clamp(self.sigma_max**2 - sigma.pow(2), min=1e-12)
        grad_pxTlxt = (x_end - x) / denom
        gt2 = 2 * sigma
        return -0.5 * gt2 * (grad_pxtlx0 - self.guidance * grad_pxTlxt)

    def _vp_snr_sqrt_reciprocal(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.exp(0.5 * self.beta_d * t.pow(2) + self.beta_min * t) - 1)

    def _vp_snr_sqrt_reciprocal_deriv(self, t: torch.Tensor) -> torch.Tensor:
        base = self._vp_snr_sqrt_reciprocal(t)
        return 0.5 * (self.beta_min + self.beta_d * t) * (base + 1 / torch.clamp(base, min=1e-12))

    def _vp_derivative(self, x: torch.Tensor, denoised: torch.Tensor, x_end: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_flat = sigma.reshape(x.shape[0])
        snr_recip = self._vp_snr_sqrt_reciprocal(sigma_flat)
        snr_recip_deriv = self._vp_snr_sqrt_reciprocal_deriv(sigma_flat)
        s_t = (1 + snr_recip.pow(2)).rsqrt()
        s_t_deriv = -snr_recip * snr_recip_deriv * s_t.pow(3)

        logs_t = self._vp_logs(sigma_flat, self.beta_d, self.beta_min)
        logsnr_t = -2 * torch.log(torch.clamp(snr_recip, min=1e-12))
        logsnr_t_max = self._vp_logsnr(torch.full_like(sigma_flat, self.sigma_max), self.beta_d, self.beta_min)
        logs_t_max = self._vp_logs(torch.full_like(sigma_flat, self.sigma_max), self.beta_d, self.beta_min)
        std_t = snr_recip * s_t

        a_t = torch.exp(logsnr_t_max - logsnr_t + logs_t - logs_t_max)
        b_t = -torch.expm1(logsnr_t_max - logsnr_t) * torch.exp(logs_t)
        mu_t = self._expand(a_t, x) * x_end + self._expand(b_t, x) * denoised
        std_t_sq = torch.clamp(self._expand(std_t, x).pow(2), min=1e-12)

        denom_q = torch.clamp(-torch.expm1(logsnr_t_max - logsnr_t), min=1e-12)
        grad_logq = -(x - mu_t) / std_t_sq / self._expand(denom_q, x)

        denom_end = torch.clamp(torch.expm1(logsnr_t - logsnr_t_max), max=-1e-12)
        grad_logpxTlxt = -(x - self._expand(torch.exp(logs_t - logs_t_max), x) * x_end) / std_t_sq / self._expand(denom_end, x)

        f = self._expand(s_t_deriv * torch.exp(-logs_t), x) * x
        gt2 = self._expand(2 * torch.exp(2 * logs_t) * snr_recip * snr_recip_deriv, x)
        return f - gt2 * (0.5 * grad_logq - self.guidance * grad_logpxTlxt)

    def _derivative(self, x: torch.Tensor, denoised: torch.Tensor, x_end: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        match self.pred_mode:
            case SDEType.VE:
                return self._ve_derivative(x, denoised, x_end, sigma)
            case SDEType.VP:
                return self._vp_derivative(x, denoised, x_end, sigma)
            case _:
                raise NotImplementedError(f"Unsupported DDBM pred_mode: {self.pred_mode}")

    def _step(self, x: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor, x_end: torch.Tensor, predicted_obj: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_end = x_end.to(device=x.device, dtype=x.dtype)
        sigma = sigma.to(device=x.device, dtype=x.dtype)
        sigma_next = sigma_next.to(device=x.device, dtype=x.dtype)
        predicted_obj = predicted_obj.to(device=x.device, dtype=x.dtype) if predicted_obj is not None else None
        sigma_hat = sigma
        x_curr = x
        if predicted_obj is None:
            _, denoised = self._denoise(x_curr, sigma, condition=x_end)
        else:
            denoised = predicted_obj

        if self.churn_step_ratio > 0 and torch.any(sigma_next > 0):
            # Optional stochastic churn matches the reference sampler's noise injection before Heun correction.
            sigma_hat = (sigma_next - sigma) * self.churn_step_ratio + sigma
            d_1 = self._derivative(x_curr, denoised, x_end, sigma)
            dt = sigma_hat - sigma
            noise_scale = torch.sqrt(torch.clamp(dt.abs(), min=0)) * torch.sqrt(torch.clamp(2 * sigma, min=0))
            x_curr = x_curr + d_1 * dt + torch.randn_like(x_curr) * noise_scale
            _, denoised = self._denoise(x_curr, sigma_hat, condition=x_end)

        d = self._derivative(x_curr, denoised, x_end, sigma_hat)
        dt = sigma_next - sigma_hat
        if torch.all(sigma_next == 0):
            return x_curr + d * dt, denoised

        # Heun's method refines the Euler proposal with a second denoiser evaluation.
        x_euler = x_curr + d * dt
        _, denoised_next = self._denoise(x_euler, sigma_next, condition=x_end)
        d_next = self._derivative(x_euler, denoised_next, x_end, sigma_next)
        x_next = x_curr + 0.5 * (d + d_next) * dt
        return x_next, denoised

    def sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.fast_sampling_steps is not None:
            i = len(self.fast_sampling_steps) - i
            tau = self.fast_sampling_steps[i]
            tau_minus_one = self.fast_sampling_steps[i + 1] if i < len(self.fast_sampling_steps) - 1 else 0
            t = torch.full((data.x.shape[0],), tau, device=data.x.device, dtype=torch.long)
            condition = data.condition.to(data.x.device) if isinstance(data.condition, torch.Tensor) else data.condition
            data = DiffusionData(data.x, t, condition=condition)
            return self.fast_sampling_step(data, tau, tau_minus_one, return_noise=return_noise, predicted_obj=predicted_obj)

        assert data.condition is not None, "Condition must be given for DDBM."
        condition = data.condition.to(data.x.device) if isinstance(data.condition, torch.Tensor) else data.condition
        assert isinstance(condition, torch.Tensor), "Condition must be given as a tensor for DDBM."
        x = condition if torch.all(data.t == self.time_steps) else data.x
        t = data.t.to(x.device)
        sigma = self._gather_sigma(t, x)
        prev_t = torch.clamp(t.long() - 1, min=0)
        sigma_next = self._gather_sigma(prev_t, x)
        predicted_obj = self.forward(DiffusionData(x, t, condition=condition)) if predicted_obj is None else predicted_obj.to(x.device)
        x_t_minus_one, denoised = self._step(x, sigma, sigma_next, condition, predicted_obj=predicted_obj)
        return (x_t_minus_one, denoised) if return_noise else x_t_minus_one

    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert data.condition is not None, "Condition must be given for DDBM."
        condition = data.condition.to(data.x.device) if isinstance(data.condition, torch.Tensor) else data.condition
        assert isinstance(condition, torch.Tensor), "Condition must be given as a tensor for DDBM."
        x = condition if tau == self.time_steps else data.x
        t = torch.full((x.shape[0],), tau, device=x.device, dtype=torch.long)
        next_t = torch.full((x.shape[0],), tau_minus_one, device=x.device, dtype=torch.long)
        sigma = self._gather_sigma(t, x)
        sigma_next = self._gather_sigma(next_t, x)
        predicted_obj = self.forward(DiffusionData(x, t, condition=condition)) if predicted_obj is None else predicted_obj.to(x.device)
        x_tau_minus_one, denoised = self._step(x, sigma, sigma_next, condition, predicted_obj=predicted_obj)
        return (x_tau_minus_one, denoised) if return_noise else x_tau_minus_one


__all__ = ["DDBMModule"]
