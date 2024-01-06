from torchmanager import losses, metrics
from torchmanager_core import torch, view
from torchmanager_core.typing import Module, Optional, Union

from diffusion.scheduling import BetaSpace

from .managers.ddpm import DDPMManager, Module


class Manager(DDPMManager[Module]):
    def __init__(self, model: Module, beta_space: BetaSpace, time_steps: int, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[losses.Loss, dict[str, losses.Loss]]] = None, metrics: dict[str, metrics.Metric] = {}) -> None:
        super().__init__(model, beta_space, time_steps, optimizer, loss_fn, metrics)
        view.warnings.warn(f"The `manager.Manager` has been deprecated from v0.2 and will be removed in v1.0. Please use `managers.DDPMManager` instead.")
