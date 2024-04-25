import abc, torch
from torchmanager_core import view
from typing import Any, Generic, Optional, Sequence, TypeVar, Union, overload

from diffusion.data import DiffusionData
from .protocols import TimedData


class TimedModule(torch.nn.Module, abc.ABC):
    """
    The basic diffusion model

    * extends: `torch.nn.Module`
    * Abstract class

    - method to implement:
        - unpack_data: The method that accepts inputs perform to `.protocols.TimedData` to unpack the given inputs and passed to `forward` method
    """

    def __call__(self, x_in: TimedData, *args: Any, **kwargs: Any) -> Any:
        data = self.unpack_data(x_in)
        return super().__call__(*data, *args, **kwargs)

    @abc.abstractmethod
    def unpack_data(self, x_in: TimedData) -> tuple[Any, ...]:
        """
        Method to unpack `TimedData`, the unpacked data will be passed as positional arguments to `forward` method

        - Parameters:
            x_in: The `TimedData` to unpack
        - Returns: A `tuple` of returned unpacked data
        """
        return NotImplemented


Module = TypeVar('Module', bound=TimedModule)


class DiffusionModule(torch.nn.Module, Generic[Module], abc.ABC):
    """
    The diffusion model that has the forward diffusion and sampling step algorithm implemented

    * extends: `torch.nn.Module`
    * Abstract class
    * Generic: `Module`

    - Properties:
        - model: The model to use for diffusion in `Module`
        - time_steps: The total time steps of diffusion model
    - method to implement:
        - forward_diffusion: The forward pass of diffusion model, sample noises
        - sampling_step: The sampling step of diffusion model
    """
    model: Module
    time_steps: int

    @property
    def sampling_range(self) -> range:
        return range(1, self.time_steps + 1)

    def __init__(self, model: Module, time_steps: int) -> None:
        super().__init__()
        self.model = model
        self.time_steps = time_steps

    def forward(self, data: DiffusionData, /) -> torch.Tensor:
        return self.model(data)

    @abc.abstractmethod
    def forward_diffusion(self, data: Any, t: torch.Tensor, /, condition: Optional[torch.Tensor] = None) -> tuple[Any, torch.Tensor]:
        """
        Forward pass of diffusion model, sample noises

        - Parameters:
            - data: Any kind of noised data
            - t: A `torch.Tensor` of the time step, sampling uniformly if not given
            - condition: An optional `torch.Tensor` of the condition to generate images
        - Returns: A `tuple` of noisy images and sampled time step in `DiffusionData` and noises in `torch.Tensor`
        """
        return NotImplemented

    @torch.no_grad()
    def sampling(self, num_images: int, x_t: torch.Tensor, *args: Any, condition: Optional[torch.Tensor] = None, sampling_range: Optional[Union[Sequence[int], range]] = None, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        '''
        Samples a given number of images

        - Parameters:
            - num_images: An `int` of number of images to generate
            - x_t: A `torch.Tensor` of the image at T step
            - condition: An optional `torch.Tensor` of the condition to generate images
            - sampling_range: An optional `Sequence[int]`, or `range` of the range of time steps to sample
            - start_index: An optional `int` of the start index of the time step
            - end_index: An `int` of the end index of the time step
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Retruns: A `list` of `torch.Tensor` generated results
        '''
        # initialize
        imgs = x_t
        sampling_range = range(self.time_steps, 0, -1) if sampling_range is None else sampling_range
        progress_bar = view.tqdm(desc='Sampling loop time step', total=len(sampling_range), disable=not show_verbose)

        # sampling loop time step
        for i in sampling_range:
            # fetch data
            t = torch.full((num_images,), i, dtype=torch.long, device=imgs.device)

            # append to predicitions
            x = DiffusionData(imgs, t, condition=condition)
            y = self.sampling_step(x, i)
            imgs = y.to(imgs.device)
            progress_bar.update()

        # reset model and loss
        return [img for img in imgs]

    @overload
    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> torch.Tensor:
        ...

    @overload
    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Sampling step of diffusion model

        - Parameters:
            - data: A `DiffusionData` object
            - i: An `int` of current time step
            - return_noise: A `bool` flag to return predicted noise
        - Returns: A `torch.Tensor` of noised image if not returning noise or a `tuple` of noised image and predicted noise in `torch.Tensor` if returning noise
        """
        return NotImplemented
