from PIL import Image
from torchmanager.callbacks import FrequencyCallback
from torchmanager.data import Dataset
from torchmanager_core import os, torch
from torchmanager_core.protocols import Frequency
from torchmanager_core.typing import Any, Generic, Optional, Sequence, TypeVar, Union

from .protocols import Samplable

S = TypeVar("S", bound=Samplable)


class SamplingCallback(FrequencyCallback, Generic[S]):
    """
    A sampling callback for Diffusion Models that samples images from given data in a given frequency of epochs.

    - Parameters:
        - sampling_data: A `Sequence` of `torch.Tensor` or `Dataset` of `torch.Tensor` for sampling.
        - sampling_dir: A `str` for saving sampled images.
        - sampling_target: A `Samplable` object for sampling.
        - frequency: An `int` for sampling frequency.
        - initial_epoch: An `int` for initial epoch.
        - sampling_device: An optional `list` of `torch.device` or `torch.device` for sampling device.
        - sampling_range: An optional `Sequence` of `int` or `range` for sampling range.
        - show_verbose: A `bool` for showing verbose.
        - use_multi_gpus: A `bool` flag of if using multi-gpus during sampling.
    """
    device: Optional[Union[list[torch.device], torch.device]]
    frequency: int
    sampling_data: Union[Sequence[torch.Tensor], Dataset[torch.Tensor]]
    sampling_dir: str
    sampling_range: Optional[Union[Sequence[int], range]]
    sampling_target: S
    show_verbose: bool
    use_multi_gpus: bool

    def __init__(self, sampling_data: Union[Sequence[torch.Tensor], Dataset[torch.Tensor]], sampling_dir: str, sampling_target: S, *, frequency: int = 20, initial_epoch: int = 0, sampling_device: Optional[Union[list[torch.device], torch.device]] = None, sampling_range: Optional[Union[Sequence[int], range]] = None, show_verbose: bool = False, use_multi_gpus: bool = False) -> None:
        """
        Construct sampling callback for Diffusion Models.
        
        - Parameters:
            - sampling_data: A `Sequence` of `torch.Tensor` or `Dataset` of `torch.Tensor` for sampling.
            - sampling_dir: A `str` for saving sampled images.
            - sampling_target: A `Samplable` object for sampling.
            - frequency: An `int` for sampling frequency.
            - initial_epoch: An `int` for initial epoch.
            - sampling_device: An optional `list` of `torch.device` or `torch.device` for sampling device.
            - sampling_range: An optional `Sequence` of `int` or `range` for sampling range.
            - show_verbose: A `bool` for showing verbose.
            - use_multi_gpus: A `bool` for using multi-gpus.
        """
        super().__init__(Frequency.EPOCH, initial_epoch)
        self.device = sampling_device
        self.frequency = frequency
        self.sampling_data = sampling_data
        self.sampling_dir = sampling_dir
        self.sampling_range = sampling_range
        self.sampling_target = sampling_target
        self.show_verbose = show_verbose
        self.use_multi_gpus = use_multi_gpus

    def _update(self, result: zip[tuple[torch.Tensor, torch.Tensor]]) -> None:
        # save sampled images
        for i, (img, sample) in enumerate(result):
            # convert to numpy
            img = img.cpu().detach().numpy()
            sample = sample.cpu().detach().numpy()

            # save as img_{i}.png and sample_{i}.png
            img_path = os.path.join(self.sampling_dir, f"img_{i}.png")
            sample_path = os.path.join(self.sampling_dir, f"sample_{i}.png")
            Image.fromarray(img).save(img_path)
            Image.fromarray(sample).save(sample_path)

    def on_epoch_end(self, epoch: int, summary: dict[str, float] = ..., val_summary: Optional[dict[str, float]] = None) -> None:
        if epoch % self.frequency == 0 and epoch != 0:
            super().on_epoch_end(epoch, summary, val_summary)

    def step(self, *args: Any, **kwargs: Any) -> zip[tuple[torch.Tensor, torch.Tensor]]:
        # initialize samples
        imgs: list[torch.Tensor] = []
        samples: list[torch.Tensor] = []

        # loop for sampling data
        for data in self.sampling_data:
            # unpack data
            if not isinstance(data, torch.Tensor):
                x = data[0]
            else:
                x = data

            # get batch size as num of images
            num_images = x.size(0)

            # predict
            sampled_imgs = self.sampling_target.predict(num_images, x.size()[1:], condition=x, sampling_range=self.sampling_range, device=self.device, empty_cache=False, use_multi_gpus=self.use_multi_gpus, show_verbose=self.show_verbose)

            # add to list
            imgs += [x]
            samples += sampled_imgs
        return zip(imgs, samples)
