"""Regression checks for inherited diffusion and optimizer interfaces."""
import argparse
import copy
import unittest

import torch
from typing import TYPE_CHECKING, assert_type

from diffusion.configs import DDPMEvalConfigs, EvalConfigs
from diffusion.data import DiffusionData
from diffusion.managers import DDPMManager, Manager
from diffusion.networks.openai import ResBlock, TimestepBlock, TimestepEmbedSequential
from diffusion.nn import DDPMModule, DiffusionModule, SDEModule
from diffusion.optim import EMAOptimizer
from diffusion.scheduling import linear_schedule
from diffusion.sde import VPSDE
from diffusion_bridges.configs import ABridgeEvalConfigs, BBDMEvalConfigs


if TYPE_CHECKING:
    def sampling_types(module: DiffusionModule[torch.nn.Module], data: DiffusionData[torch.Tensor], flag: bool) -> None:
        assert_type(module.sampling_step(data, 1), torch.Tensor)
        assert_type(module.sampling_step(data, 1, return_noise=True), tuple[torch.Tensor, torch.Tensor])
        assert_type(module.sampling_step(data, 1, return_noise=flag), torch.Tensor | tuple[torch.Tensor, torch.Tensor])


class InterfaceTests(unittest.TestCase):
    def test_optimizer_forwards_zero_grad_and_closure(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = EMAOptimizer(torch.optim.SGD([parameter], lr=0.1), [parameter])
        parameter.sum().backward()
        optimizer.zero_grad(set_to_none=False)
        assert parameter.grad is not None
        self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter)))
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(parameter.grad)
        self.assertIsNone(optimizer.step())
        self.assertEqual(optimizer.step(lambda: 2.0), 2.0)
        restored = copy.deepcopy(optimizer)
        self.assertEqual(restored.ema_decay, optimizer.ema_decay)
        self.assertFalse(restored.is_ema_parameters)

    def test_eval_configs_preserve_parser_and_keyword_defaults(self) -> None:
        for config in [EvalConfigs, DDPMEvalConfigs, BBDMEvalConfigs, ABridgeEvalConfigs]:
            with self.subTest(config=config.__name__):
                parser = argparse.ArgumentParser()
                returned = config.get_arguments(parser, batch_size=7, device='cpu', model_help='Model path')
                self.assertIs(returned, parser)
                args = parser.parse_args(['data', 'model'])
                self.assertEqual(args.batch_size, 7)
                self.assertEqual(args.device, 'cpu')
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group('Diffusion')
        self.assertIs(EvalConfigs.get_arguments(group), group)

    def test_timestep_sequential_remains_a_timestep_block(self) -> None:
        block = ResBlock(32, 64, dropout=0)
        sequential = TimestepEmbedSequential(TimestepEmbedSequential(block))
        self.assertIsInstance(sequential, TimestepBlock)
        x, embedding = torch.randn(2, 32, 4, 4), torch.randn(2, 64)
        with torch.no_grad():
            result = sequential.forward(input=x, emb=embedding)
        self.assertEqual(result.shape, x.shape)

    def test_manager_property_and_selected_backward_inputs(self) -> None:
        network = torch.nn.Linear(2, 2)
        module = DDPMModule(network, linear_schedule(10), 10)
        manager = Manager(module)
        manager.time_steps = 8
        self.assertEqual(module.time_steps, 8)
        output = network(torch.ones(1, 2)).sum()
        manager.backward(output, inputs=[network.weight])
        self.assertIsNotNone(network.weight.grad)
        self.assertIsNone(network.bias.grad)
        with self.assertWarns(DeprecationWarning):
            legacy_manager = DDPMManager(network, linear_schedule(10), 10)
        self.assertEqual(legacy_manager.time_steps, 10)
        vars(legacy_manager)["time_steps"] = 6
        self.assertEqual(legacy_manager.time_steps, 6)

    def test_sde_supports_positional_time_and_precomputed_score(self) -> None:
        beta_space = linear_schedule(10)
        module = SDEModule(torch.nn.Identity(), VPSDE(10, beta_space=beta_space), 10, beta_space=beta_space)
        x = torch.zeros(2, 3, 4, 4)
        t = torch.tensor([0.2, 0.7])
        data, target = module.forward_diffusion(x, t, noise=torch.ones_like(x))
        self.assertTrue(torch.equal(data.t, t))
        self.assertEqual(target.shape, x.shape)
        score = torch.ones_like(x)
        result, returned_score = module.sampling_step(data, 1, predicted_obj=score, return_noise=True)
        self.assertIs(returned_score, score)
        self.assertEqual(result.shape, x.shape)
