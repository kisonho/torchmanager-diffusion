import torch, unittest
from typing import cast


class _EchoTimedModule(torch.nn.Module):
    last_t: torch.Tensor | None

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = value
        self.last_t = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.last_t = t.detach().clone()
        return torch.full_like(x, self.value)


class _LargeConditionalModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return torch.full_like(x, 100.0)


class Case0103(unittest.TestCase):
    def test_import(self):
        import diffusion

        try:
            from packaging.version import Version # type: ignore
        except ImportError:
            return

        self.assertGreaterEqual(diffusion.VERSION, Version("v1.3a1"))

    def test_module(self):
        from diffusion.data.diffusion import DiffusionData
        from diffusion.networks import build
        from diffusion_bridges import BBDMModule

        # build model
        unet = build(3, 3, dim_mults=(1, 2, 4, 8))
        T = 1000
        model = BBDMModule(unet, T)

        # initialize testing data
        x = torch.rand((4, 3, 256, 256))
        t = torch.randint(1, T + 1, (x.shape[0],), device=x.device).long()
        xt = DiffusionData(x, t)

        # pass to model
        y: torch.Tensor = model(xt)
        self.assertEqual(y.shape, x.shape, f"Output shape ({y.shape}) mismatch with input shape ({x.shape}).")

    def test_sde_module_forward_matches_score_sde_label_conventions(self) -> None:
        from diffusion.data import DiffusionData
        from diffusion.nn import SDEModule
        from diffusion.scheduling import linear_schedule
        from diffusion.sde import VESDE, VPSDE

        steps = 10
        x = torch.zeros((2, 3, 4, 4))
        beta_space = linear_schedule(steps)

        vp_model = _EchoTimedModule()
        vp_module = SDEModule(vp_model, VPSDE(steps, beta_space=beta_space), steps, beta_space=beta_space)
        vp_t = torch.tensor([0.5, 0.9])
        vp_score = vp_module(DiffusionData(x, vp_t))
        vp_labels = vp_t * (steps - 1)
        vp_std = beta_space.sqrt_one_minus_alphas_cumprod[vp_labels.long()]
        expected_vp = -torch.ones_like(x) / vp_std[:, None, None, None]
        self.assertTrue(torch.allclose(vp_score, expected_vp))
        assert vp_model.last_t is not None
        self.assertTrue(torch.allclose(vp_model.last_t, vp_labels))

        ve_model = _EchoTimedModule()
        ve_module = SDEModule(ve_model, VESDE(steps), steps)
        ve_t = torch.tensor([0.25, 0.75])
        ve_score = ve_module(DiffusionData(x, ve_t))
        self.assertTrue(torch.allclose(ve_score, torch.ones_like(x)))
        assert ve_model.last_t is not None
        expected_ve_labels = ((1 - ve_t) * (steps - 1)).round().long()
        self.assertTrue(torch.equal(ve_model.last_t.long(), expected_ve_labels))

    def test_sde_forward_diffusion_returns_score_target(self) -> None:
        from diffusion.nn import SDEModule
        from diffusion.scheduling import linear_schedule
        from diffusion.sde import VPSDE

        steps = 10
        x = torch.randn((2, 3, 4, 4))
        t = torch.tensor([0.2, 0.7])
        beta_space = linear_schedule(steps)
        module = SDEModule(_EchoTimedModule(), VPSDE(steps, beta_space=beta_space), steps, beta_space=beta_space)

        xt, objective = module.forward_diffusion(x, t=t)
        mean, std = module.sde.marginal_prob(x, t)
        expected = -(xt.x - mean) / std[:, None, None, None].pow(2)
        self.assertTrue(torch.allclose(objective, expected, atol=1e-5, rtol=1e-4))

    def test_subvp_sampling_uses_reverse_diffusion_predictor(self) -> None:
        from diffusion.data import DiffusionData
        from diffusion.nn import SDEModule
        from diffusion.scheduling import linear_schedule
        from diffusion.sde import SubVPSDE

        steps = 10
        beta_space = linear_schedule(steps)
        module = SDEModule(_EchoTimedModule(value=0.0), SubVPSDE(steps, beta_space=beta_space), steps, beta_space=beta_space)
        x = torch.randn((2, 3, 4, 4))
        t = torch.tensor([0.4, 0.8])

        torch.manual_seed(1234)
        sampled = cast(torch.Tensor, module.sampling_step(DiffusionData(x, t), 1))

        torch.manual_seed(1234)
        f, G = module.sde.discretize(x, t)
        expected = x - f + G[:, None, None, None] * torch.randn_like(x)
        self.assertTrue(torch.allclose(sampled, expected))

    def test_ddbm_vp_derivative_matches_reference_endpoint_guidance(self) -> None:
        from diffusion.sde import SDEType
        from diffusion_bridges import DDBMModule

        module = DDBMModule(_EchoTimedModule(), 10, pred_mode=SDEType.VP)
        x = torch.tensor([[[[0.2, -0.4]]], [[[0.7, -0.1]]]])
        denoised = torch.tensor([[[[-0.3, 0.5]]], [[[0.1, -0.6]]]])
        x_end = torch.tensor([[[[0.4, -0.2]]], [[[-0.5, 0.8]]]])
        sigma = torch.tensor([0.25, 0.75])[:, None, None, None]

        actual = module._vp_derivative(x, denoised, x_end, sigma)

        sigma_flat = sigma.reshape(x.shape[0])
        sigma_t = module._vp_snr_sqrt_reciprocal(sigma_flat)
        sigma_t_deriv = module._vp_snr_sqrt_reciprocal_deriv(sigma_flat)
        s_t = (1 + sigma_t.pow(2)).rsqrt()
        s_t_deriv = -sigma_t * sigma_t_deriv * s_t.pow(3)
        std_t = sigma_t * s_t
        logs_t = module._vp_logs(sigma_flat, module.beta_d, module.beta_min)
        logs_T = module._vp_logs(torch.ones_like(sigma_flat), module.beta_d, module.beta_min)
        logsnr_t = -2 * torch.log(sigma_t)
        logsnr_T = module._vp_logsnr(torch.ones_like(sigma_flat), module.beta_d, module.beta_min)

        def expand(values: torch.Tensor) -> torch.Tensor:
            return values[:, None, None, None]

        a_t = torch.exp(logsnr_T - logsnr_t + logs_t - logs_T)
        b_t = -torch.expm1(logsnr_T - logsnr_t) * torch.exp(logs_t)
        mu_t = expand(a_t) * x_end + expand(b_t) * denoised
        std_t_sq = expand(std_t).pow(2)
        grad_logq = -(x - mu_t) / std_t_sq / expand(-torch.expm1(logsnr_T - logsnr_t))
        grad_logpxTlxt = -(x - expand(torch.exp(logs_t - logs_T)) * x_end) / std_t_sq / expand(torch.expm1(logsnr_t - logsnr_T))
        f = expand(s_t_deriv * torch.exp(-logs_t)) * x
        gt2 = expand(2 * torch.exp(2 * logs_t) * sigma_t * sigma_t_deriv)
        expected = f - gt2 * (0.5 * grad_logq - module.guidance * grad_logpxTlxt)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_ddbm_sampling_schedule_matches_reference_endpoint(self) -> None:
        from diffusion.sde import SDEType
        from diffusion_bridges import DDBMModule

        module = DDBMModule(_EchoTimedModule(), 40, sigma_min=0.0001, sigma_max=1.0, pred_mode=SDEType.VP)
        x = torch.zeros((2, 1, 1, 1))
        t = torch.full((2,), module.time_steps, dtype=torch.long)

        train_sigma = module._gather_sigma(t, x)
        sample_sigma = module._gather_sampling_sigma(t, x)

        self.assertTrue(torch.allclose(train_sigma, torch.ones_like(train_sigma)))
        self.assertTrue(torch.allclose(sample_sigma, torch.full_like(sample_sigma, 0.9999)))

    def test_ddbm_sampling_schedule_is_backward_compatible_with_pickled_modules(self) -> None:
        from diffusion.sde import SDEType
        from diffusion_bridges import DDBMModule

        module = DDBMModule(_EchoTimedModule(), 40, sigma_min=0.0001, sigma_max=1.0, pred_mode=SDEType.VP)
        del module._buffers["sampling_sigma_schedule"]
        x = torch.zeros((2, 1, 1, 1))
        t = torch.full((2,), module.time_steps, dtype=torch.long)

        sample_sigma = module._gather_sampling_sigma(t, x)

        self.assertIn("sampling_sigma_schedule", module._buffers)
        self.assertTrue(torch.allclose(sample_sigma, torch.full_like(sample_sigma, 0.9999)))
        self.assertNotIn("sampling_sigma_schedule", module.state_dict())

    def test_ddbm_sampling_clips_denoised_and_final_sample(self) -> None:
        from diffusion.sde import SDEType
        from diffusion_bridges import DDBMModule

        module = DDBMModule(_LargeConditionalModule(), 4, sigma_min=0.0001, sigma_max=1.0, pred_mode=SDEType.VP)
        x = torch.zeros((2, 1, 2, 2))
        condition = torch.zeros_like(x)
        t = torch.ones((x.shape[0],), dtype=torch.long)

        sampled, denoised = module.sampling_step(DiffusionData(x, t, condition=condition), 4, return_noise=True)

        self.assertLessEqual(float(denoised.max()), 1.0)
        self.assertGreaterEqual(float(denoised.min()), -1.0)
        self.assertLessEqual(float(sampled.max()), 1.0)
        self.assertGreaterEqual(float(sampled.min()), -1.0)
