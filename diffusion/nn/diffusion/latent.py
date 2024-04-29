import abc, torch
from enum import Enum
from torchmanager_core import view
from typing import Any, Generic, Optional, Sequence, TypeVar, Union, overload

from diffusion.data import DiffusionData
from .diffusion import DiffusionModule, TimedModule

Module = TypeVar('Module', bound=TimedModule)
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

    def __call__(self, *args: Any, latent_mode: LatentMode = LatentMode.FORWARD, **kwds: Any) -> Any:
        return super().__call__(*args, latent_mode=latent_mode, **kwds)

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

    @torch.no_grad()
    def fast_sampling(self, num_images: int, x_t: torch.Tensor, sampling_range: Sequence[int], condition: Optional[torch.Tensor] = None, *, show_verbose: bool = False) -> list[torch.Tensor]:
        '''
        Samples a given number of images using fast sampling algorithm.

        - Parameters:
            - num_images: An `int` of number of images to generate
            - x_t: A `torch.Tensor` of the image at T step
            - sampling_range: An `Iterable[int]` of the range of time steps to sample
            - condition: An optional `torch.Tensor` of the condition to generate images
            - start_index: An optional `int` of the start index of reversed time step
            - end_index: An `int` of the end index of reversed time step
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Retruns: A `list` of `torch.Tensor` generated results
        '''
        # initialize
        progress_bar = view.tqdm(desc='Sampling loop time step', total=len(sampling_range)) if show_verbose else None

        # sampling loop time step
        for i, tau in enumerate(sampling_range):
            # fetch data
            t = torch.full((num_images,), tau, dtype=torch.long, device=x_t.device)
            tau_minus_one = sampling_range[i+1] if i < len(sampling_range) - 1 else 0

            # append to predicitions
            x = DiffusionData(x_t, t, condition=condition)
            y = self.fast_sampling_step(x, tau, tau_minus_one)
            assert isinstance(y, torch.Tensor), "The output must be a valid `torch.Tensor`."
            x_t = y.to(x_t.device)

            # update progress bar
            if progress_bar is not None:
                progress_bar.update()

        # return final image
        x_0 = x_t
        return [img for img in x_0]

    def forward(self, *args: Any, latent_mode: LatentMode = LatentMode.FORWARD, **kwargs: Any) -> Any:
        if latent_mode == LatentMode.ENCODE:
            return self.encode(*args, **kwargs)
        elif latent_mode == LatentMode.DECODE:
            return self.decode(*args, **kwargs)
        elif latent_mode == LatentMode.FORWARD:
            return super().forward(*args, **kwargs)
        else:
            raise NotImplementedError(f"Mode {latent_mode} is not implemented for diffusion model")

    def sampling(self, num_images: int, x_t: torch.Tensor, /, *, condition: Optional[torch.Tensor] = None, fast_sampling: bool = False, sampling_range: Optional[Union[Sequence[int], range]] = None, show_verbose: bool = False) -> list[torch.Tensor]:
        '''
        Samples a given number of images

        - Parameters:
            - num_images: An `int` of number of images to generate
            - x_t: A `torch.Tensor` of the image at T step
            - condition: An optional `torch.Tensor` of the condition to generate images
            - start_index: An optional `int` of the start index of reversed time step
            - end_index: An `int` of the end index of reversed time step
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Retruns: A `list` of `torch.Tensor` generated results
        '''
        # enter latent space
        assert condition is not None, 'Condition is required for sampling.'
        print(self.encoder)
        z_t = self(x_t, mode=LatentMode.ENCODE)
        z_condition = self(condition, mode=LatentMode.ENCODE)

        # sampling
        if fast_sampling:
            assert sampling_range is not None, 'Sampling range is required for fast sampling.'
            sampling_steps = list(sampling_range)
            z_0_list = self.fast_sampling(num_images, z_t, sampling_steps, condition=z_condition, show_verbose=show_verbose)
        else:
            assert not isinstance(sampling_range, list), 'Sampling range must be a `range` or `reversed` for original sampling.'
            z_0_list = super().sampling(num_images, z_t, condition=z_condition, sampling_range=sampling_range, show_verbose=show_verbose)

        # exit latent space
        z_0_list = [img.unsqueeze(0) for img in z_0_list]
        z_0 = torch.cat(z_0_list, dim=0)
        x_0 = self(z_0, mode=LatentMode.DECODE)
        return [img for img in x_0]
