import abc, torch
from enum import Enum
from typing import Any, Generic, TypeVar, overload

from .diffusion import DiffusionModule
from .ddpm import DDPMModule
from .protocols import BetaSpace

Module = TypeVar('Module', bound=torch.nn.Module)
E = TypeVar('E', bound=torch.nn.Module | None)
D = TypeVar('D', bound=torch.nn.Module | None)


class LatentMode(Enum):
    """
    The enumeration of the latent forward mode

    * extends: `Enum`
    """
    ENCODE = 'encode'
    DECODE = 'decode'
    FORWARD = 'forward'


class LatentDiffusionModule(DiffusionModule[Module], Generic[Module, E, D], abc.ABC):
    """
    The diffusion model that has the forward diffusion and sampling step algorithm implemented with latent space

    * extends: `DiffusionModule`
    * Abstract class
    * Generic: `E`, `Module`, `D`

    - Properties:
        - encoder: The encoder model in `E`
        - decoder: The decoder model in `D`
    - method to implement:
        - forward_diffusion: The forward pass of diffusion model, sample noises
        - sampling_step: The sampling step of diffusion model
    """
    encoder: E
    decoder: D

    def __init__(self, model: Module, time_steps: int, /, *, encoder: E = None, decoder: D = None) -> None:
        super().__init__(model, time_steps)
        # initialize encoder
        self.encoder = encoder
        if self.encoder is not None:
            self.encoder.eval()

        # initialize decoder
        self.decoder = decoder
        if self.decoder is not None:
            self.decoder.eval()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            return z
        return self.decoder(z)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return x
        return self.encoder(x)

    @overload
    def __call__(self, x_in: Any, *args, mode: LatentMode = LatentMode.FORWARD, **kwargs) -> Any:
        ...

    @overload
    def __call__(self, x_in: torch.Tensor, mode: LatentMode = LatentMode.DECODE) -> torch.Tensor:
        ...

    @overload
    def __call__(self, x_in: torch.Tensor, mode: LatentMode = LatentMode.ENCODE) -> torch.Tensor:
        ...

    def __call__(self, *args, mode: LatentMode = LatentMode.FORWARD, **kwargs) -> Any:
        if mode == LatentMode.ENCODE:
            return self.encode(*args, **kwargs)
        elif mode == LatentMode.DECODE:
            return self.decode(*args, **kwargs)
        elif mode == LatentMode.FORWARD:
            return super().__call__(*args, **kwargs)


class LDM(LatentDiffusionModule[Module, E, D], DDPMModule[Module]):
    """
    Latent DDPM module.
    """
    def __init__(self, model: Module, beta_space: BetaSpace, time_steps: int, /, *, encoder: E = None, decoder: D = None, with_condition: bool = False) -> None:
        LatentDiffusionModule.__init__(self, model, time_steps, encoder=encoder, decoder=decoder)
        DDPMModule.__init__(self, model, beta_space, time_steps, with_condition=with_condition)


__all__ = ["LatentMode", "LatentDiffusionModule", "LDM"]
