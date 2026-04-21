import torch, unittest


class _EchoTimedModule(torch.nn.Module):
    last_t: torch.Tensor | None

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = value
        self.last_t = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.last_t = t.detach().clone()
        return torch.full_like(x, self.value)


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
        from diffusion.nn import BBDMModule
        from diffusion.scheduling import linear_schedule

        # build model
        unet = build(3, 3, dim_mults=(1, 2, 4, 8))
        T = 1000
        linear_beta_space = linear_schedule(T)
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
        sampled = module.sampling_step(DiffusionData(x, t), i=1)

        torch.manual_seed(1234)
        f, G = module.sde.discretize(x, t)
        expected = x - f + G[:, None, None, None] * torch.randn_like(x)
        self.assertTrue(torch.allclose(sampled, expected))
