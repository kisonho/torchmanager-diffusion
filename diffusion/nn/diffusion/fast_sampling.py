import abc, torch
from typing import TypeVar, overload

from .diffusion import DiffusionModule
from .protocols import DiffusionData

Module = TypeVar('Module', bound=torch.nn.Module)


class FastSamplingDiffusionModule(DiffusionModule[Module], abc.ABC):
    """
    The diffusion model that supports fast sampling

    * extends: `DiffusionModule`
    * Abstract class
    * Generic: `Module`

    - Properties:
        - `fast_sampling_steps`: An optional `list` of fast sampling steps for the diffusion process in `int`
        - `fast_sampling`: A `bool` flag of if enables fast sampling during the diffusion process
    - method to implement:
        - `fast_sampling_step`: Samples a single time step using fast sampling algorithm.
    """
    fast_sampling_steps: list[int] | None

    @property
    def fast_sampling(self) -> bool:
        return self.fast_sampling_steps is not None

    def __init__(self, model: Module, time_steps: int) -> None:
        super().__init__(model, time_steps)
        self.fast_sampling_steps = None

    @overload
    @abc.abstractmethod
    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: torch.Tensor | None = None) -> torch.Tensor:
        ...

    @overload
    @abc.abstractmethod
    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = True, predicted_obj: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abc.abstractmethod
    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''
        Samples a single time step using fast sampling algorithm.

        - Parameters:
            - data: A `DiffusionData` of the data to sample
            - tau: An `int` of the current time step
            - tau_minus_one: An `int` of the next time step
            - return_noise: A `bool` flag to return noise
            - predicted_obj: An optional `torch.Tensor` of the predicted object
        - Returns: A `torch.Tensor` of the sampled image or a `tuple` of `torch.Tensor` of the sampled image and `torch.Tensor` of the noise
        '''
        ...
