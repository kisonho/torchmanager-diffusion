import argparse, diffusion, os
from diffusion.configs import TrainingConfigs
from diffusion.sde import SDEType
from diffusion.version import DESCRIPTION
from torchmanager_core import view


class BBDMTrainingConfigs(TrainingConfigs):
    vqgan_path: str | None

    def format_arguments(self) -> None:
        super().format_arguments()
        self.vqgan_path = os.path.normpath(self.vqgan_path) if self.vqgan_path is not None else None

    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup = argparse.ArgumentParser()) -> argparse.ArgumentParser | argparse._ArgumentGroup:
        parser = TrainingConfigs.get_arguments(parser)
        sde_bbdm_args = parser.add_argument_group("BBDM Arguments")
        sde_bbdm_args.add_argument("-vq", "--vqgan_path", type=str, default=None, help="The path for the VQGAN model if using latent space.")
        return parser

    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"BBDM settings: vqgan_path={self.vqgan_path}")


class ABridgeTrainingConfigs(BBDMTrainingConfigs):
    c_lambda: float

    def format_arguments(self) -> None:
        super().format_arguments()
        assert self.c_lambda > 0, "Lambda must be a positive number."

    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup = argparse.ArgumentParser()) -> argparse.ArgumentParser | argparse._ArgumentGroup:
        parser = BBDMTrainingConfigs.get_arguments(parser)
        sde_bbdm_args = parser.add_argument_group("A-Bridge Arguments")
        sde_bbdm_args.add_argument("-l", "--c_lambda", type=float, default=2, help="The lambda for the loss function, default is 2.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"diffusion={diffusion.VERSION}")

    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"A-Bridge settings: c_lambda={self.c_lambda}")


class DDBMTrainingConfigs(BBDMTrainingConfigs):
    sigma_data: float
    sigma_min: float
    sigma_max: float
    rho: float
    beta_d: float
    beta_min: float
    cov_xy: float
    guidance: float
    churn_step_ratio: float
    pred_mode: SDEType

    def format_arguments(self) -> None:
        super().format_arguments()
        self.pred_mode = self.pred_mode if isinstance(self.pred_mode, SDEType) else SDEType[str(self.pred_mode).upper()]
        assert self.sigma_data > 0, "Sigma data must be a positive number."
        assert self.sigma_min > 0, "Sigma min must be a positive number."
        assert self.sigma_max > self.sigma_min, "Sigma max must be greater than sigma min."
        assert self.rho > 0, "Rho must be a positive number."
        assert self.beta_d > 0, "Beta d must be a positive number."
        assert self.beta_min > 0, "Beta min must be a positive number."
        assert self.guidance >= 0, "Guidance must be a non-negative number."
        assert self.churn_step_ratio >= 0, "Churn step ratio must be a non-negative number."
        self.pred_mode = SDEType(self.pred_mode)

    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup = argparse.ArgumentParser()) -> argparse.ArgumentParser | argparse._ArgumentGroup:
        parser = BBDMTrainingConfigs.get_arguments(parser)
        ddbm_args = parser.add_argument_group("DDBM Arguments")
        ddbm_args.add_argument("--sigma_data", type=float, default=0.5, help="The data standard deviation, default is 0.5.")
        ddbm_args.add_argument("--sigma_min", type=float, default=0.002, help="The minimum bridge time, default is 0.002.")
        ddbm_args.add_argument("--sigma_max", type=float, default=1.0, help="The maximum bridge time, default is 1.0.")
        ddbm_args.add_argument("--rho", type=float, default=7.0, help="The Karras schedule exponent, default is 7.0.")
        ddbm_args.add_argument("--beta_d", type=float, default=2.0, help="The VP beta coefficient, default is 2.0.")
        ddbm_args.add_argument("--beta_min", type=float, default=0.1, help="The VP minimum beta, default is 0.1.")
        ddbm_args.add_argument("--cov_xy", type=float, default=0.0, help="The covariance term between source and target endpoints, default is 0.0.")
        ddbm_args.add_argument("--guidance", type=float, default=1.0, help="The bridge guidance coefficient, default is 1.0.")
        ddbm_args.add_argument("--churn_step_ratio", type=float, default=0.0, help="The stochastic churn ratio, default is 0.0.")
        ddbm_args.add_argument("--pred_mode", type=str, default="vp", choices=("ve", "vp"), help="The bridge schedule mode, default is `vp`.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"diffusion={diffusion.VERSION}")

    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(
            "DDBM settings: "
            f"sigma_data={self.sigma_data}, sigma_min={self.sigma_min}, sigma_max={self.sigma_max}, "
            f"rho={self.rho}, beta_d={self.beta_d}, beta_min={self.beta_min}, cov_xy={self.cov_xy}, "
            f"guidance={self.guidance}, churn_step_ratio={self.churn_step_ratio}, pred_mode={self.pred_mode.name}"
        )


class I2SBTrainingConfigs(BBDMTrainingConfigs):
    beta_max: float
    ot_ode: bool

    def format_arguments(self) -> None:
        super().format_arguments()
        assert self.beta_max > 0, "Beta max must be a positive number."

    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser | argparse._ArgumentGroup = argparse.ArgumentParser()) -> argparse.ArgumentParser | argparse._ArgumentGroup:
        parser = BBDMTrainingConfigs.get_arguments(parser)
        i2sb_args = parser.add_argument_group("I2SB Arguments")
        i2sb_args.add_argument("--beta_max", type=float, default=0.3, help="The maximum diffusion coefficient, default is 0.3.")
        i2sb_args.add_argument("--ot_ode", action="store_true", help="Use deterministic OT-ODE updates for I2SB.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"diffusion={diffusion.VERSION}")

    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"I2SB settings: beta_max={self.beta_max}, ot_ode={self.ot_ode}")


__all__ = ["ABridgeTrainingConfigs", "BBDMTrainingConfigs", "DDBMTrainingConfigs", "I2SBTrainingConfigs"]
