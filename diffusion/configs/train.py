from torchmanager.configs import Configs as _Configs
from torchmanager_core import argparse, os, torch, view, _raise, VERSION as tm_version
from torchmanager_core.typing import Optional, Union

from diffusion.scheduling import BetaScheduler
from diffusion.sde import SDEType
from diffusion.version import DESCRIPTION


class Configs(_Configs):
    """Basic Training Configurations"""
    batch_size: int
    ckpt_path: Optional[str]
    data_dir: str
    devices: Optional[list[torch.device]]
    epochs: int
    output_model: str
    show_verbose: bool
    time_steps: int
    use_multi_gpus: bool

    @property
    def default_device(self) -> Optional[torch.device]:
        return None if self.devices is None else self.devices[0]

    def format_arguments(self) -> None:
        # format arguments
        super().format_arguments()
        self.ckpt_path = os.path.normpath(self.ckpt_path) if self.ckpt_path is not None else None
        self.data_dir = os.path.normpath(self.data_dir)
        self.output_model = os.path.normpath(self.output_model)

        # initialize device
        if isinstance(self.devices, list):
            self.devices = [torch.device(device) for device in self.devices]

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

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # experiment arguments
        parser.add_argument("data_dir", type=str, help="The dataset directory.")
        parser.add_argument("output_model", type=str, help="The path for the final PyTorch model.")

        # training arguments
        training_args = parser.add_argument_group("Training Arguments")
        training_args.add_argument("-b", "--batch_size", type=int, default=64, help="The batch size, default is 64.")
        training_args.add_argument("-e", "--epochs", type=int, default=100, help="The training epochs, default is 100.")
        training_args.add_argument("-t", "--time_steps", type=int, default=1000, help="The total time steps of diffusion model, default is 1000.")
        training_args.add_argument("-d", "--devices", type=str, default=None, nargs="+", help="The device(s) used for training, default is `None`.")
        training_args.add_argument("--ckpt_path", type=str, default=None, help="The path to the checkpoint file to continue training.")
        training_args.add_argument("--show_verbose", action="store_true", default=False, help="A flag to show verbose.")
        training_args = _Configs.get_arguments(training_args)

        # device arguments
        device_args = parser.add_argument_group("Device Arguments")
        
        device_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="A flag to use multiple GPUs during training.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"torchmanager={tm_version}")

    def show_settings(self) -> None:
        view.logger.info(f"Data directory: {self.data_dir}")
        view.logger.info(f"Output model: {self.output_model}")
        view.logger.info(f"Training settings: batch_size={self.batch_size}, epoch={self.epochs}, show_verbose={self.show_verbose}")
        view.logger.info(f"Device settings: device={self.devices}, use_multi_gpus={self.use_multi_gpus}")
        view.logger.info(f"Diffusion model settings: time_steps={self.time_steps}")
        if self.ckpt_path is not None:
            view.logger.info(f"From checkpoint: {self.ckpt_path}")


class DDPMTrainingConfigs(Configs):
    """Training Configurations for DDPM."""
    beta_range: Optional[list[float]]
    beta_scheduler: BetaScheduler

    def format_arguments(self) -> None:
        # format arguments
        self.beta_scheduler = BetaScheduler(self.beta_scheduler)

        # check beta range format
        if self.beta_range is not None:
            assert len(self.beta_range) == 2, "Beta range must be a two-sized list."
            assert self.beta_range[0] > 0 and self.beta_range[1] > 0, "Beta start and end must be all positive numbers."

        # format super arguments
        super().format_arguments()

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # experiment arguments
        parser = Configs.get_arguments(parser)

        # diffusion arguments
        diffusion_args = parser.add_argument_group("DDPM Arguments")
        diffusion_args.add_argument("-beta", "--beta_scheduler", type=str, default="linear", help="The beta scheduler for diffusion model, default is 'linear'.")
        diffusion_args.add_argument("--beta_range", type=float, default=None, nargs=2, help="The range of mid-linear scheduler, default is `None`.")
        return parser

    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"DDPM settings: beta_scheduler={self.beta_scheduler}, beta_range={self.beta_range}")


class SDETrainingConfigs(Configs):
    """Training Configurations for SDE."""
    sde_type: SDEType

    def format_arguments(self) -> None:
        self.sde_type = self.sde_type if isinstance(self.sde_type, SDEType) else SDEType[str(self.sde_type).upper()]
        super().format_arguments()

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # experiment arguments
        parser = Configs.get_arguments(parser)

        # diffusion arguments
        diffusion_args = parser.add_argument_group("SDE Arguments")
        diffusion_args.add_argument("-sde", "--sde_type", type=str, default="VE", help="The type of SDE, default is 'VE'.")
        return parser

    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"SDE settings: sde_type={self.sde_type.name}")
