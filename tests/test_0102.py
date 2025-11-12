import torch, unittest


class Case0102(unittest.TestCase):
    def test_ema(self):
        from diffusion.optim import EMAOptimizer
        from torch.optim import AdamW

        linear = torch.nn.Linear(10, 10)
        adamw = AdamW(linear.parameters())
        ema = EMAOptimizer(adamw, linear.parameters())
        self.assertIsInstance(ema, EMAOptimizer)
        self.assertEqual(ema.base_optimizer, adamw)

    def test_import(self):
        import diffusion

        try:
            from packaging.version import Version # type: ignore
        except ImportError:
            return

        self.assertGreaterEqual(diffusion.VERSION, Version("v1.2"))
