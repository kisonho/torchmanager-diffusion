import torch, unittest


class Case0101(unittest.TestCase):
    def test_import(self):
        import diffusion

        try:
            from packaging.version import Version # type: ignore
        except ImportError:
            return

        self.assertGreaterEqual(diffusion.VERSION, Version("v1.1"))

    def test_module(self):
        from diffusion.data.diffusion import DiffusionData
        from diffusion.networks import build
        from diffusion.nn import DDPM
        from diffusion.scheduling import linear_schedule

        # build model
        unet = build(3, 3, dim_mults=(1, 2, 4, 8))
        T = 1000
        linear_beta_space = linear_schedule(T)
        model = DDPM(unet, linear_beta_space, T)

        # initialize testing data
        x = torch.rand((4, 3, 256, 256))
        t = torch.randint(1, T + 1, (x.shape[0],), device=x.device).long()
        xt = DiffusionData(x, t)

        # pass to model
        y: torch.Tensor = model(xt)
        self.assertEqual(y.shape, x.shape, f"Output shape ({y.shape}) mismatch with input shape ({x.shape}).")
