import torch
from torchmanager import callbacks, losses

import data
import diffusion
from diffusion.configs import TrainingConfigs


def train(configs: TrainingConfigs, /) -> diffusion.networks.Unet:
    """
    Train a diffusion model with `diffusion.configs.TrainingConfigs`

    - Parameters:
        - configs: A `diffusion.configs.TrainingConfigs` for training configs
    - Returns: A trained `diffusion.networks.Unet` model
    """
    # load datasets
    training_dataset, validation_dataset, in_channels, _ = data.Datasets(configs.dataset).load(configs.data_dir, configs.batch_size, device=configs.device)

    # load model
    model = diffusion.networks.load_unet(in_channels)

    # load optimizer, loss, and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    mse = losses.MSE()

    # calculate betas
    T = configs.time_steps
    if configs.beta_range is not None and configs.beta_scheduler == diffusion.scheduling.BetaScheduler.PEAK_LINEAR:
        beta_lower, beta_peak = configs.beta_range
        beta_space = diffusion.scheduling.peak_linear_schedule(T, beta_lower=beta_lower, beta_peak=beta_peak)
    elif configs.beta_range is not None:
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
    model = manager.fit(training_dataset, configs.epochs, device=configs.device, use_multi_gpus=configs.use_multi_gpus, val_dataset=validation_dataset, show_verbose=configs.show_verbose, callbacks_list=callbacks_list)

    # save model
    torch.save(model, configs.output_model)
    return model


if __name__ == "__main__":
    configs = TrainingConfigs.from_arguments()
    train(configs)
