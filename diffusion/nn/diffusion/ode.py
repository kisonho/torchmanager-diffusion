import abc, torch
from typing import TypeVar, overload

from diffusion.data.diffusion import DiffusionData

from .diffusion import DiffusionModule

Module = TypeVar('Module', bound=torch.nn.Module)


class ODESamplingDiffusionModule(DiffusionModule[Module], abc.ABC):
    """
    A diffusion module that can switch between deterministic ODE and stochastic SDE sampling.

    * extends: `DiffusionModule`
    * Abstract class
    * Generic: `Module`

    - Properties:
        - use_ode: A `bool` flag to choose ODE sampling when `True` and SDE sampling when `False`
    - methods to implement:
    """
    __use_ode: bool

    @property
    def use_ode(self) -> bool:
        return self.__use_ode if hasattr(self, "_ODESamplingDiffusionModule__use_ode") else False

    @use_ode.setter
    def use_ode(self, use_ode: bool):
        self.__use_ode = use_ode

    def __init__(self, model: Module, time_steps: int, *, use_ode: bool = False) -> None:
        super().__init__(model, time_steps)
        self.use_ode = use_ode

    @overload
    def sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None) -> torch.Tensor:
        ...

    @overload
    def sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = False) -> torch.Tensor:
        ...

    @overload
    def sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.ode_sampling_step(data, i, predicted_obj=predicted_obj, return_noise=return_noise) if self.use_ode else self.sde_sampling_step(data, i, predicted_obj=predicted_obj, return_noise=return_noise)

    @overload
    @abc.abstractmethod
    def ode_sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None) -> torch.Tensor:
        ...

    @overload
    @abc.abstractmethod
    def ode_sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = False) -> torch.Tensor:
        ...

    @overload
    @abc.abstractmethod
    def ode_sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abc.abstractmethod
    def ode_sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic ODE sampling step of diffusion model.

        - Parameters:
            - data: A `DiffusionData` object
            - i: An `int` of current time step
            - predicted_obj: An optional `torch.Tensor` of a precomputed model prediction
            - return_noise: A `bool` flag to return predicted noise
        - Returns: A `torch.Tensor` of sampled data if not returning noise or a `tuple` of sampled data and predicted noise in `torch.Tensor` if returning noise
        """
        ...

    @overload
    @abc.abstractmethod
    def sde_sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None) -> torch.Tensor:
        ...

    @overload
    @abc.abstractmethod
    def sde_sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = False) -> torch.Tensor:
        ...

    @overload
    @abc.abstractmethod
    def sde_sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abc.abstractmethod
    def sde_sampling_step(self, data: DiffusionData, i: int, /, *, predicted_obj: torch.Tensor | None = None, return_noise: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Stochastic SDE sampling step of diffusion model.

        - Parameters:
            - data: A `DiffusionData` object
            - i: An `int` of current time step
            - predicted_obj: An optional `torch.Tensor` of a precomputed model prediction
            - return_noise: A `bool` flag to return predicted noise
        - Returns: A `torch.Tensor` of sampled data if not returning noise or a `tuple` of sampled data and predicted noise in `torch.Tensor` if returning noise
        """
        ...


__all__ = ["ODESamplingDiffusionModule"]
