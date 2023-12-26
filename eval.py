import ssl
import torch
from torchmanager import metrics
from torchmanager_core import devices, view
from torchmanager_core.typing import Optional
from torchvision import models

import data
import diffusion
from diffusion import DDPMManager as Manager
from diffusion.configs import DDPMEvalConfigs as TestingConfigs


def eval(configs: TestingConfigs, /, model: Optional[torch.nn.Module] = None) -> dict[str, float]:
    """
    Test with `diffusion.configs.TestingConfigs`

    - Parameters:
        - model: An optional pre-trained `torch.nn.Module`
        - configs: A `diffusion.configs.TestingConfigs` for testing
    - Returns: A `dict` of results with name as `str` and value as `float`
    """
    # initialize FID
    ssl._create_default_https_context = ssl._create_unverified_context
    inception = models.inception_v3(pretrained=True)
    inception.fc = torch.nn.Identity()  # type: ignore
    inception.eval()
    fid = metrics.FID(inception)

    # load checkpoint
    if configs.model is not None and configs.model.endswith(".model"):
        # load checkpoint
        manager: Manager[torch.nn.Module] = Manager.from_checkpoint(configs.model, map_location=devices.CPU)  # type: ignore

        # set time steps
        if configs.time_steps is not None:
            manager.time_steps = configs.time_steps

        # set beta scheduler
        if configs.beta_scheduler is not None:
            beta_space = diffusion.scheduling.BetaScheduler(configs.beta_scheduler).calculate_space(manager.time_steps)
            manager.beta_space = beta_space

        # set metrics
        manager.metric_fns = {
            "FID": fid,
        }
    else:
        # load beta space
        assert model is not None or configs.model is not None, "Either pre-trained model should be given as parameters or its path has been given in configurations."
        model = torch.load(configs.model, map_location=devices.CPU) if configs.model is not None else model
        assert isinstance(model, torch.nn.Module), "The pre-trained model is not a valid PyTorch model or torchmanager checkpoint."
        assert configs.beta_scheduler is not None, "Beta scheduler is required when loading a PyTorch model."
        assert configs.time_steps is not None, "Time steps is required when loading a PyTorch model."
        beta_space = diffusion.scheduling.BetaScheduler(configs.beta_scheduler).calculate_space(configs.time_steps)
        manager = Manager(model, beta_space, configs.time_steps, metrics={"FID": fid})  # type: ignore

    # load dataset
    dataset, _, _, _ = data.load_cifar10(configs.data_dir, configs.batch_size)

    # evaluation
    result = manager.test(dataset, sampling_images=True, device=configs.device, use_multi_gpus=configs.use_multi_gpus, show_verbose=configs.show_verbose)
    return result


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()
    assert isinstance(configs, TestingConfigs)
    result = eval(configs)
    view.logger.info(result)
