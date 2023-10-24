import unittest


class Case0100(unittest.TestCase):
    def test_configs(self):
        import os
        from diffusion.configs import TrainingConfigs as Configs

        configs = Configs.from_arguments(*[
            "~/Downloads/Dataset/",
            "tests/case0100/test.pth",
            "-exp", "case_0100.exp",
            "--replace_experiment",
        ])
        assert isinstance(configs, Configs), "The `configs` is not a valid `diffusion.configs.TrainingConfigs`."

        self.assertIsNone(configs.dataset)
        self.assertEqual(configs.data_dir, os.path.normpath("~/Downloads/Dataset/"))
        self.assertEqual(configs.output_model, os.path.normpath("tests/case0100/test.pth"))
        self.assertEqual(configs.experiment, "case_0100.exp")

    def test_import(self):
        import data, diffusion

        try:
            from packaging.version import Version # type: ignore
            self.assertGreaterEqual(diffusion.VERSION, Version("v0.1a"))
        except ImportError:
            pass

    def test_scheduler(self) -> None:
        import torch
        from diffusion.scheduling import BetaScheduler, BetaSpace

        T = 1000
        beta_scheduler = BetaScheduler("linear")
        beta_space = beta_scheduler.calculate_space(T)
        correct_beta_space = BetaSpace(torch.linspace(0.0001, 0.01, T))
        eq_betas = beta_space.betas - correct_beta_space.betas
        eq_betas = eq_betas.sum()
        self.assertEqual(eq_betas, 0)
