import argparse
import torchmanager
from torchmanager.configs import Configs as _Configs
from torchmanager_core import argparse, os, torch, view, _raise
from torchmanager_core.typing import Optional, Union

from ..version import DESCRIPTION


class DDPMEvalConfigs(_Configs):
    """Training Configurations"""
    batch_size: int
    beta_scheduler: Optional[str]
    data_dir: str
    dataset: str
    device: torch.device
    image_size: int
    model: str
    show_verbose: bool
    time_steps: Optional[int]
    use_multi_gpus: bool

    def format_arguments(self) -> None:
        # format arguments
        super().format_arguments()
        self.data_dir = os.path.normpath(self.data_dir)
        self.device = torch.device(self.device)
        self.model = os.path.normpath(self.model)

        # assert formats
        assert self.batch_size > 0, _raise(ValueError(f"Batch size must be a positive number, got {self.batch_size}."))
        assert self.image_size > 0, _raise(ValueError(f"Image size must be a positive number, got {self.image_size}."))
        if self.time_steps is not None:
            assert self.time_steps > 0, _raise(ValueError(f"Time steps must be a positive number, got {self.time_steps}."))

        # format logging
        formatter = view.logging.Formatter("%(message)s")
        console = view.logging.StreamHandler()
        console.setLevel(view.logging.INFO)
        console.setFormatter(formatter)
        view.logger.addHandler(console)

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # experiment arguments
        parser.add_argument("data_dir", type=str, help="The dataset directory.")
        parser.add_argument("model", type=str, help="The path for a pre-trained PyTorch model, default is `None`.")

        # training arguments
        testing_args = parser.add_argument_group("Testing Arguments")
        testing_args.add_argument("-b", "--batch_size", type=int, default=1, help="The batch size, default is 1.")
        testing_args.add_argument("-beta", "--beta_scheduler", type=str, default=None, help="The beta scheduler for diffusion model, default is 'None' (Checkpoint is needed).")
        testing_args.add_argument("--dataset", type=str, default=None, help="The target type of dataset.")
        testing_args.add_argument("-size", "--image_size", type=int, default=32, help="The image size to generate, default is 32.")
        testing_args.add_argument("--show_verbose", action="store_true", default=False, help="A flag to show verbose.")
        testing_args.add_argument("-t", "--time_steps", type=int, default=None, help="The total time steps of diffusion model, default is `None` (Checkpoint is needed).")
        testing_args = _Configs.get_arguments(testing_args)

        # device arguments
        device_args = parser.add_argument_group("Device Arguments")
        device_args.add_argument("--device", type=str, default="cuda", help="The target device to run for the experiment.")
        device_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="A flag to use multiple GPUs during training.")
        return parser
    
    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"torchmanager={torchmanager.version}")

    def show_settings(self) -> None:
        view.logger.info(f"Dataset {self.dataset}: {self.data_dir}")
        view.logger.info(f"Model: {self.model}")
        view.logger.info(f"Testing settings: batch_size={self.batch_size}, show_verbose={self.show_verbose}")
        view.logger.info(f"Diffusion model settings: beta_scheduler={self.beta_scheduler}, time_steps={self.time_steps}")
        view.logger.info(f"Device settings: device={self.device}, use_multi_gpus={self.use_multi_gpus}")
