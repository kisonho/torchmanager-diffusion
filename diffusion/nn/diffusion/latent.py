import abc, torch
from enum import Enum
from typing import Any, Generic, Optional, TypeVar, Union, overload

from diffusion.data import DiffusionData
from diffusion.nn.diffusion.protocols import TimedData
from .diffusion import DiffusionModule

Module = TypeVar('Module', bound=torch.nn.Module)
E = TypeVar('E', bound=Optional[torch.nn.Module])
D = TypeVar('D', bound=Optional[torch.nn.Module])


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
    fast_sampling_steps: Optional[list[int]]

    @property
    def fast_sampling(self) -> bool:
        return self.fast_sampling_steps is not None

    def __init__(self, model: Module, time_steps: int, /, *, encoder: E = None, decoder: D = None) -> None:
        super().__init__(model, time_steps)
        self.fast_sampling_steps = None

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
    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...

    @overload
    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = True, predicted_obj: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        raise NotImplementedError('Fast sampling step method has not been implemented yet.')

    @overload
    def __call__(self, x_in: Any, mode: LatentMode = LatentMode.FORWARD) -> Any:
        ...

    @overload
    def __call__(self, x_in: torch.Tensor, mode: LatentMode = LatentMode.DECODE) -> torch.Tensor:
        ...

    @overload
    def __call__(self, x_in: torch.Tensor, mode: LatentMode = LatentMode.ENCODE) -> torch.Tensor:
        ...

    def __call__(self, x_in: Any, mode: LatentMode = LatentMode.FORWARD) -> Any:
        if mode == LatentMode.ENCODE:
            assert isinstance(x_in, torch.Tensor), f'Input data must be a `torch.Tensor` to encode, got {type(x_in)}.'
            return self.encode(x_in)
        elif mode == LatentMode.DECODE:
            assert isinstance(x_in, torch.Tensor), f'Input data must be a `torch.Tensor` to decode, got {type(x_in)}.'
            return self.decode(x_in)
        elif mode == LatentMode.FORWARD:
            return super().__call__(x_in)
        else:
            raise NotImplementedError(f'Latent forward mode {mode} is not implemented.')
