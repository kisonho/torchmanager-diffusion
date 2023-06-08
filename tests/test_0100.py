import unittest


class Case0100(unittest.TestCase):
    def test_configs(self):
        import os
        from diffusion.configs import TrainingConfigs as Configs

        configs = Configs.from_arguments(*[
            "mnist",
            "~/Downloads/Dataset/",
            "tests/case0100/test.pth",
            "-exp", "case_0100.exp",
            "--replace_experiment",
        ])

        self.assertEqual(configs.dataset, "mnist")
        self.assertEqual(configs.data_dir, os.path.normpath("~/Downloads/Dataset/"))
        self.assertEqual(configs.output_model, os.path.normpath("tests/case0100/test.pth"))
        self.assertEqual(configs.experiment, "case_0100.exp")

    def test_data(self):
        import math
        from data import Datasets
        from torchmanager_core import devices

        root_dir = "~/Downloads/Datasets/"
        batch_size = 32

        training_dataset, testing_dataset, in_channels, _ = Datasets.MNIST.load(root_dir, batch_size, device=devices.CPU)

        self.assertEqual(training_dataset.unbatched_len, 60000)
        self.assertEqual(training_dataset.batched_len, int(60000 / 32))
        self.assertEqual(testing_dataset.unbatched_len, 10000)
        self.assertEqual(testing_dataset.batched_len, math.ceil(10000 / 32))
        self.assertEqual(in_channels, 1)

        for x in training_dataset:
            self.assertEqual(x.shape, (batch_size, 1, 28, 28))

        for x in testing_dataset:
            self.assertEqual(x.shape[1:], (1, 28, 28))
            self.assertLessEqual(x.shape[0], batch_size)

    def test_import(self):
        import data, diffusion

        try:
            from packaging.version import Version # type: ignore
            self.assertGreaterEqual(diffusion.VERSION, Version("v0.1a"))
        except ImportError:
            pass

    def test_harmonic_schedule(self) -> None:
        import time, torch
        from diffusion.scheduling import BetaScheduler

        beta_space = BetaScheduler.HARMONIC.calculate_space(1000)
        beta_summation = beta_space.betas.sum()
        self.assertEqual(beta_space.betas.shape[0], 1000)
        self.assertGreaterEqual(beta_summation, 5) # type: ignore

        time_steps = 300
        N = 10000

        start = time.time()
        harmonic_space_alg1_values = torch.zeros(time_steps, dtype=torch.float32)
        harmonic_space_alg1 = harmonic_space_alg1_values
        for i in range(time_steps):
            harmonic_space_alg1_values[i] = 1.05 ** i
            harmonic_space_alg1 = (harmonic_space_alg1_values - torch.min(harmonic_space_alg1_values)).round() / (harmonic_space_alg1_values.max() - harmonic_space_alg1_values.min()) * N
            harmonic_space_alg1 = harmonic_space_alg1 / N
        time_cost_alg1 = time.time() - start

        start = time.time()
        harmonic_space_alg2 = torch.tensor([N * (1.05 ** i) for i in range(time_steps)])
        harmonic_space_alg2 = (harmonic_space_alg2 - torch.min(harmonic_space_alg2)).round() / (harmonic_space_alg2.max() - harmonic_space_alg2.min())
        time_cost_alg2 = time.time() - start

        self.assertLessEqual(time_cost_alg2, time_cost_alg1)

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
