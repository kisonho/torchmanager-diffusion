import torchmanager
from torchmanager.configs import Configs as _Configs
from torchmanager_core import argparse, os, torch, view, _raise
from torchmanager_core.typing import Optional, Union

from diffusion.scheduling import BetaScheduler
from diffusion.version import DESCRIPTION


class Configs(_Configs):
    """Training Configurations"""
    batch_size: int
    beta_range: Optional[list[float]]
    beta_scheduler: BetaScheduler
    data_dir: str
    dataset: Optional[str]
    device: Optional[torch.device]
    epochs: int
    output_model: str
    show_verbose: bool
    time_steps: int
    use_multi_gpus: bool

    def format_arguments(self) -> None:
        # format arguments
        super().format_arguments()
        self.beta_scheduler = BetaScheduler(self.beta_scheduler)
        self.data_dir = os.path.normpath(self.data_dir)
        self.device = torch.device(self.device) if self.device is not None else None
        self.output_model = os.path.normpath(self.output_model)

        # initialize console
        if self.show_verbose:
            formatter = view.logging.Formatter("%(message)s")
            console = view.logging.StreamHandler()
            console.setLevel(view.logging.INFO)
            console.setFormatter(formatter)
            view.logger.addHandler(console)

        # assert formats
        assert self.batch_size > 0, _raise(ValueError(f"Batch size must be a positive number, got {self.batch_size}."))
        assert self.epochs > 0, _raise(ValueError(f"Epochs must be a positive number, got {self.epochs}."))
        assert self.time_steps > 0, _raise(ValueError(f"Time steps must be a positive number, got {self.time_steps}."))

        # check beta range format
        if self.beta_range is not None:
            assert len(self.beta_range) == 2, "Beta range must be a two-sized list."
            assert self.beta_range[0] > 0 and self.beta_range[1] > 0, "Beta start and end must be all positive numbers."

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # experiment arguments
        parser.add_argument("data_dir", type=str, help="The dataset directory.")
        parser.add_argument("output_model", type=str, help="The path for the final PyTorch model.")

        # training arguments
        training_args = parser.add_argument_group("Training Arguments")
        training_args.add_argument("-d", "--dataset", type=str, default=None, help="The target type of dataset.")
        training_args.add_argument("-b", "--batch_size", type=int, default=64, help="The batch size, default is 64.")
        training_args.add_argument("-e", "--epochs", type=int, default=100, help="The training epochs, default is 100.")
        training_args.add_argument("--show_verbose", action="store_true", default=False, help="A flag to show verbose.")

        # diffusion arguments
        diffusion_args = parser.add_argument_group("DDPM Arguments")
        diffusion_args.add_argument("-beta", "--beta_scheduler", type=str, default="linear", help="The beta scheduler for diffusion model, default is 'linear'.")
        diffusion_args.add_argument("--beta_range", type=float, default=None, nargs=2, help="The range of mid-linear scheduler, default is `None`.")
        diffusion_args.add_argument("-t", "--time_steps", type=int, default=1000, help="The total time steps of diffusion model, default is 1000.")
        diffusion_args = _Configs.get_arguments(training_args)

        # device arguments
        device_args = parser.add_argument_group("Device Arguments")
        device_args.add_argument("--device", type=str, default=None, help="The target device to run for the experiment.")
        device_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="A flag to use multiple GPUs during training.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"torchmanager={torchmanager.version}")

    def show_settings(self) -> None:
        view.logger.info(f"Dataset {self.dataset}: {self.data_dir}")
        view.logger.info(f"Output Model: {self.output_model}")
        view.logger.info(f"Training settings: batch_size={self.batch_size}, epoch={self.epochs}, show_verbose={self.show_verbose}")
        view.logger.info(f"Diffusion model settings: beta_scheduler={self.beta_scheduler}, beta_range={self.beta_range}, time_steps={self.time_steps}")
        view.logger.info(f"Device settings: device={self.device}, use_multi_gpus={self.use_multi_gpus}")
