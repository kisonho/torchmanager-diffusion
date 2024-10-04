from torchmanager import callbacks, losses
from torchmanager_core import argparse, torch, view
from torchmanager_core.typing import Union

import data, diffusion
from diffusion.configs import DDPMTrainingConfigs


class TrainingConfigs(DDPMTrainingConfigs):
    dataset: data.Datasets

    def format_arguments(self) -> None:
        super().format_arguments()
        self.dataset = data.Datasets(self.dataset)

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        parser.add_argument("--dataset", type=str, help="The dataset used for training")
        return DDPMTrainingConfigs.get_arguments(parser)

    def show_settings(self) -> None:
        view.logger.info(f"Dataset: {self.dataset}")
        super().show_settings()


def train(configs: TrainingConfigs, /) -> diffusion.networks.Unet:
    """
    Train a diffusion model with `diffusion.configs.TrainingConfigs`

    - Parameters:
        - configs: A `diffusion.configs.TrainingConfigs` for training configs
    - Returns: A trained `diffusion.networks.Unet` model
    """
    # load datasets
    training_dataset, validation_dataset, in_channels, _ = configs.dataset.load(configs.data_dir, configs.batch_size, device=configs.default_device)

    # load model
    model = diffusion.networks.build_unet(in_channels)

    # load optimizer, loss, and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    mse = losses.MSE()

    # calculate betas
    T = configs.time_steps
    if configs.beta_range is not None:
        beta_start, beta_end = configs.beta_range
        beta_space = configs.beta_scheduler.calculate_space_with_range(T, beta_start=beta_start, beta_end=beta_end)
    else:
        beta_space = configs.beta_scheduler.calculate_space(T)

    # initialize manager
    manager = diffusion.DDPMManager(model, beta_space, T, optimizer=optimizer, loss_fn=mse)

    # initialize callbacks
    experiment_callback = callbacks.Experiment(configs.experiment, manager, monitors={"loss": callbacks.MonitorType.MIN})
    callbacks_list: list[callbacks.Callback] = [experiment_callback]

    # train model
    model = manager.fit(training_dataset, configs.epochs, device=configs.devices, use_multi_gpus=configs.use_multi_gpus, val_dataset=validation_dataset, show_verbose=configs.show_verbose, callbacks_list=callbacks_list)
    assert isinstance(model, torch.nn.Module), "The model returned from manager is not a valid `torch.nn.Module`"

    # save model
    torch.save(model, configs.output_model)
    return model


if __name__ == "__main__":
    configs = TrainingConfigs.from_arguments()
    assert isinstance(configs, TrainingConfigs), "The configs fetched from arguments is not a valid `TrainingConfigs`"
    train(configs)
