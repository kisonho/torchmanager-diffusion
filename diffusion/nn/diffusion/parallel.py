import torch
from torch.nn import DataParallel
from typing import Any, Optional, Sequence, Union, TypeVar, overload

from diffusion.data import DiffusionData
from .diffusion import DiffusionModule

Module = TypeVar('Module', bound=DiffusionModule)


class DiffusionDataParallel(DataParallel[Module]):
    """
    The parallel model that has the forward diffusion and sampling step algorithm implemented

    * extends: `DataParallel`
    * Generic: `Module`

    - Properties:
        - time_steps: The number of time steps in diffusion model in `int`
    """
    @property
    def time_steps(self) -> int:
        return self.module.time_steps

    @time_steps.setter
    def time_steps(self, time_steps: int) -> None:
        self.module.time_steps = time_steps

    def forward_diffusion(self, x: torch.Tensor, t: torch.Tensor, *, condition: Optional[torch.Tensor] = None) -> tuple[Any, torch.Tensor]:
        return self.module.forward_diffusion(x, t, condition=condition)
    
    def sampling(self, num_images: int, x_t: torch.Tensor, *args: Any, condition: Optional[torch.Tensor] = None, sampling_range: Optional[Union[Sequence[int], range]] = None, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        return self.module.sampling(num_images, x_t, *args, condition=condition, sampling_range=sampling_range, show_verbose=show_verbose, **kwargs)

    @overload
    def sampling_step(self, data: DiffusionData, i: int, /) -> torch.Tensor:
        ...

    @overload
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.module.sampling_step(data, i, return_noise=return_noise)
