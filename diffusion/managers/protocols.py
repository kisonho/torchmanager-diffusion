import abc, torch
from diffusion.data import DiffusionData
from typing import Any, Optional, Protocol, Sequence, Union, overload


class Diffusable(Protocol):
    @property
    def time_steps(self) -> int:
        ...

    @time_steps.setter
    def time_steps(self, time_steps: int) -> None:
        ...

    def forward_diffusion(self, data: Any, t: torch.Tensor, *,condition: Optional[Any] = None) -> tuple[Any, Any]:
        ...

    def sampling(self, num_images: int, x_t: torch.Tensor, *args: Any, condition: Optional[torch.Tensor] = None, sampling_range: Optional[Union[Sequence[int], range]] = None, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        ...

    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ...
