import torch.nn.functional as F
from torchmanager.data import sliding_window
from torchmanager.metrics import MeanIoU
from torchmanager_core import torch
from torchmanager_core.typing import Callable, Generic, Module, Optional


class MIoU(MeanIoU, Generic[Module]):
    """
    The mean intersection over union metric (mIoU) for image generation using a segmentation network.

    - Parameters:
        - normalize_fn: An optional `Callable` function that accepts an image in `torch.Tensor` and returns a normalized image in `torch.Tensor`.
        - segmentation_network: A `torch.nn.Module` to segment the generated images.
    """
    normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
    segmentation_network: Module

    def __init__(self, segmentation_network: Module, /, dim: int = 1, target: Optional[str] = None, *, normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        '''
        The mean intersection over union metric (mIoU) for image generation using a segmentation network.

        - Parameters:
            - segmentation_network: A `torch.nn.Module` to segment the generated images.
            - dim: The dimension in `int` to calculate the metric.
            - target: The target key in `str` to calculate the metric.
            - normalize_fn: An optional `Callable` function that accepts an image in `torch.Tensor` and returns a normalized image in `torch.Tensor`.
        '''
        super().__init__(dim, target=target)
        self.normalize_fn = normalize_fn
        self.segmentation_network = segmentation_network

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # segment input
        with torch.no_grad():
            # denormalize input
            img = input / 2
            img += 0.5
            img = img.clip(0,1)

            # normalize image
            img = self.normalize_fn(img) if self.normalize_fn is not None else img

            # upsampling
            resized_imgs = F.interpolate(input, size=target.shape[-2:], mode='bilinear', align_corners=False)

            # initialize windows
            windowed_imgs: list[torch.Tensor] = []
            windowed_targets: list[torch.Tensor] = []

            # sliding windows
            for img, t in zip(resized_imgs, target):
                windowed_imgs.append(sliding_window(img, (768, 768), (512, 512)))
                windowed_targets.append(sliding_window(t, (768, 768), (512, 512)))

            # concat windows
            x = torch.cat(windowed_imgs)
            y = torch.cat(windowed_targets)

            # segmentation
            x = self.segmentation_network(x)
            return super().forward(x, y)
