from typing import Any, overload
from typing_extensions import override
import argparse, diffusion, os
from diffusion.configs import EvalConfigs
from diffusion.version import DESCRIPTION
from torchmanager_core import view


class BBDMEvalConfigs(EvalConfigs):
    """Evaluation Configurations"""
    vqgan_path: str | None

    @override
    def format_arguments(self) -> None:
        super().format_arguments()
        self.vqgan_path = os.path.normpath(self.vqgan_path) if self.vqgan_path is not None else None

    @staticmethod
    @overload
    def get_arguments(parser: argparse.ArgumentParser = ..., **kwargs: Any) -> argparse.ArgumentParser:
        ...

    @staticmethod
    @overload
    def get_arguments(parser: argparse._ArgumentGroup, **kwargs: Any) -> argparse._ArgumentGroup:
        ...

    @staticmethod
    @override
    def get_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup = argparse.ArgumentParser(), **kwargs: Any) -> argparse.ArgumentParser | argparse._ArgumentGroup:
        parser = EvalConfigs.get_arguments(parser, **kwargs)

        # BBDM arguments
        bbdm_args = parser if isinstance(parser, argparse._ArgumentGroup) else parser.add_argument_group("BBDM Arguments")
        bbdm_args.add_argument("-vq", "--vqgan_path", type=str, default=None, help="The path for the VQGAN model.")
        return parser

    @override
    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"diffusion={diffusion.VERSION}")

    @override
    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"BBDM settings: vqgan_path={self.vqgan_path}")


class ABridgeEvalConfigs(BBDMEvalConfigs):
    c_lambda: float | None

    @override
    def format_arguments(self) -> None:
        super().format_arguments()
        if self.c_lambda is not None:
            assert self.c_lambda > 0, "Lambda must be a positive number."

    @staticmethod
    @overload
    def get_arguments(parser: argparse.ArgumentParser = ..., **kwargs: Any) -> argparse.ArgumentParser:
        ...

    @staticmethod
    @overload
    def get_arguments(parser: argparse._ArgumentGroup, **kwargs: Any) -> argparse._ArgumentGroup:
        ...

    @staticmethod
    @override
    def get_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup = argparse.ArgumentParser(), **kwargs: Any) -> argparse.ArgumentParser | argparse._ArgumentGroup:
        parser = BBDMEvalConfigs.get_arguments(parser, **kwargs)
        abridge_args = parser if isinstance(parser, argparse._ArgumentGroup) else parser.add_argument_group("A-Bridge Arguments")
        abridge_args.add_argument("-l", "--c_lambda", type=float, default=None, help="The lambda for the loss function, default is `None`.")
        return parser

    @override
    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"A-Bridge settings: c_lambda={self.c_lambda}")


__all__ = ["ABridgeEvalConfigs", "BBDMEvalConfigs"]
