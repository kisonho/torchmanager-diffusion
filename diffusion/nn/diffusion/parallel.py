import torch
from enum import Enum
from torch.nn import DataParallel
from typing import Any, Generic, Optional, Sequence, Union, TypeVar, overload

from diffusion.data import DiffusionData
from .diffusion import DiffusionModule

Module = TypeVar('Module', bound=DiffusionModule)


class DiffusionSamplingDirection(Enum):
    FORWARD = 'forward'
    BACKWARD = 'backward'


class DiffusionSamplingModule(torch.nn.Module, Generic[Module]):
    model: Module

    @overload
    def forward(self, data: Any, t: torch.Tensor, /, *, condition: Optional[torch.Tensor] = None, direction: DiffusionSamplingDirection = DiffusionSamplingDirection.FORWARD) -> tuple[Any, torch.Tensor]:
        ...

    @overload
    def forward(self, data: DiffusionData, t: int, /, *, direction: DiffusionSamplingDirection = DiffusionSamplingDirection.BACKWARD) -> torch.Tensor:
        ...

    @overload
    def forward(self, data: DiffusionData, t: int, /, *, return_noise: bool = True, direction: DiffusionSamplingDirection = DiffusionSamplingDirection.BACKWARD) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(self, data: Any, t: Union[int, torch.Tensor], *args: Any, direction: DiffusionSamplingDirection = DiffusionSamplingDirection.BACKWARD, **kwargs: Any) -> Any:
        if direction == DiffusionSamplingDirection.FORWARD:
            assert isinstance(t, torch.Tensor), 'The forward diffusion requires the time step as `torch.Tensor`.'
            return self.model.forward_diffusion(data, t, *args, **kwargs)
        elif direction == DiffusionSamplingDirection.BACKWARD:
            assert isinstance(t, int), 'The backward diffusion requires the time step as `int`.'
            return self.model.sampling_step(data, t, *args, **kwargs)


class DiffusionDataParallel(DataParallel[DiffusionSamplingModule[Module]]):
    """
    The parallel model that has the forward diffusion and sampling step algorithm implemented

    * extends: `DataParallel`
    * Generic: `Module`
    * Implements: `managers.protocols.Diffusable`

    - Properties:
        - time_steps: The number of time steps in diffusion model in `int`
    """
    @property
    def raw_module(self) -> Module:
        return self.module.model

    @property
    def time_steps(self) -> int:
        return self.raw_module.time_steps

    @time_steps.setter
    def time_steps(self, time_steps: int) -> None:
        self.raw_module.time_steps = time_steps

    def __init__(self, module: DiffusionSamplingModule[Module], device_ids: Optional[Sequence[Union[int, torch.device]]] = None, output_device: Optional[Union[int, torch.device]] = None, dim: int = 0) -> None:
        super().__init__(module, device_ids, output_device, dim)

    def forward_diffusion(self, data: Any, t: torch.Tensor, *, condition: Optional[torch.Tensor] = None) -> tuple[Any, torch.Tensor]:
        return self(data, t, condition=condition, direction=DiffusionSamplingDirection.FORWARD)

    @overload
    def sampling_step(self, data: DiffusionData, i: int, /) -> torch.Tensor:
        ...

    @overload
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self(data, i, return_noise=return_noise, direction=DiffusionSamplingDirection.BACKWARD)
