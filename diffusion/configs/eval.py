import torchmanager
from torchmanager.configs import Configs as _Configs
from torchmanager_core import argparse, os, torch, view, raise_error

from .protocols import DESCRIPTION


class EvalConfigs(_Configs):
    """Basic Evaluation Configurations"""
    batch_size: int
    data_dir: str
    device: torch.device | None
    fast_sampling: bool
    model: str
    show_verbose: bool
    time_steps: int | None
    use_multi_gpus: bool
    use_ode: bool

    def format_arguments(self) -> None:
        # format arguments
        super().format_arguments()
        self.data_dir = os.path.normpath(self.data_dir)
        self.device = torch.device(self.device) if self.device is not None else None
        self.model = os.path.normpath(self.model)

        # assert formats
        assert self.batch_size > 0, raise_error(ValueError(f"Batch size must be a positive number, got {self.batch_size}."))
        if self.time_steps is not None:
            assert self.time_steps > 0, raise_error(ValueError(f"Time steps must be a positive number, got {self.time_steps}."))

        # format logging
        formatter = view.logging.Formatter("%(message)s")
        console = view.logging.StreamHandler()
        console.setLevel(view.logging.INFO)
        console.setFormatter(formatter)
        view.logger.addHandler(console)

    @staticmethod
    def get_arguments(
        parser: argparse.ArgumentParser | argparse._ArgumentGroup = argparse.ArgumentParser(),
        *,
        batch_size: int = 64,
        device: str | None = None,
        model_help: str = "The path for a pre-trained PyTorch model or a torchmanager checkpoint, default is `None`.",
    ) -> argparse.ArgumentParser | argparse._ArgumentGroup:
        # experiment arguments
        parser.add_argument("data_dir", type=str, help="The dataset directory.")
        parser.add_argument("model", type=str, help=model_help)

        # testing arguments
        testing_args = parser.add_argument_group("Testing Arguments")
        testing_args.add_argument("-b", "--batch_size", type=int, default=batch_size, help=f"The batch size, default is {batch_size}.")
        testing_args.add_argument("--fast_sampling", action="store_true", default=False, help="A flag to use fast sampling.")
        testing_args.add_argument("--show_verbose", action="store_true", default=False, help="A flag to show verbose.")
        testing_args.add_argument("-t", "--time_steps", type=int, default=None, help="The total time steps of diffusion model, default is `None` (Checkpoint is needed).")
        testing_args.add_argument("--use_ode", action="store_true", default=False, help="A flag to use ODE sampling.")
        testing_args = _Configs.get_arguments(testing_args)

        # device arguments
        device_args = parser.add_argument_group("Device Arguments")
        device_args.add_argument("--device", type=str, default=device, help="The target device to run for the experiment.")
        device_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="A flag to use multiple GPUs during training.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"torchmanager={torchmanager.version}")

    def show_settings(self) -> None:
        view.logger.info(f"Data directory: {self.data_dir}")
        view.logger.info(f"Pre-trained model: {self.model}")
        view.logger.info(f"Testing settings: batch_size={self.batch_size}, show_verbose={self.show_verbose}")
        view.logger.info(f"Diffusion model settings: fast_sampling={self.fast_sampling}, time_steps={self.time_steps}, use_ode={self.use_ode}")
        view.logger.info(f"Device settings: device={self.device}, use_multi_gpus={self.use_multi_gpus}")


class DDPMEvalConfigs(EvalConfigs):
    """Training Configurations"""
    beta_scheduler: str | None
    image_size: int

    def format_arguments(self) -> None:
        super().format_arguments()
        assert self.image_size > 0, raise_error(ValueError(f"Image size must be a positive number, got {self.image_size}."))

    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup = argparse.ArgumentParser()) -> argparse.ArgumentParser | argparse._ArgumentGroup:
        parser = EvalConfigs.get_arguments(
            parser,
            batch_size=1,
            device="cuda",
            model_help="The path for a pre-trained PyTorch model, default is `None`.",
        )

        # DDPM arguments
        ddpm_args = parser.add_argument_group("DDPM Arguments")
        ddpm_args.add_argument("-beta", "--beta_scheduler", type=str, default=None, help="The beta scheduler for diffusion model, default is 'None' (Checkpoint is needed).")
        ddpm_args.add_argument("-size", "--image_size", type=int, default=32, help="The image size to generate, default is 32.")
        return parser

    def show_settings(self) -> None:
        view.logger.info(f"Data directory: {self.data_dir}")
        view.logger.info(f"Model: {self.model}")
        view.logger.info(f"Testing settings: batch_size={self.batch_size}, show_verbose={self.show_verbose}")
        view.logger.info(f"Diffusion model settings: beta_scheduler={self.beta_scheduler}, time_steps={self.time_steps}, use_ode={self.use_ode}")
        view.logger.info(f"Device settings: device={self.device}, use_multi_gpus={self.use_multi_gpus}")


__all__ = ["EvalConfigs", "DDPMEvalConfigs"]
