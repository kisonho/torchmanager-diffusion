from numpy import isin
from torchmanager.metrics import Metric
from torchmanager_core import torch
from torchmanager_core.typing import Any
from typing import Optional


class Diversity(Metric):
    __sample_dim: int

    @property
    def sample_dim(self) -> int:
        return self.__sample_dim

    @sample_dim.setter
    def sample_dim(self, value: int) -> None:
        assert value >= 0, f"Dimension must be a non-negative integer, got {value}."
        self.__sample_dim = value

    def __init__(self, sample_dim: int = 1, target: Optional[str] = None) -> None:
        super().__init__(target=target)
        self.sample_dim = sample_dim

    def forward(self, input: torch.Tensor, target: Any) -> torch.Tensor:
        # denormalize input
        imgs = input / 2
        imgs += 0.5
        imgs = imgs.clip(0,1)
        imgs *= 255
        # return imgs.std(dim=self.sample_dim).mean()

        # permute images to make sample dimension as dim 0
        imgs = imgs.permute(self.sample_dim, *range(self.sample_dim), *range(self.sample_dim+1, imgs.ndim))

        # calculate mean
        img_mean = torch.zeros_like(imgs[0])
        for j in range(len(imgs)):
            img_mean = img_mean + imgs[j]
        img_mean = img_mean / len(imgs)

        # calculate std
        img_var = torch.zeros_like(imgs[0])
        for j in range(len(imgs)):
            img_var = img_var + (imgs[j] - img_mean)**2
        img_var = img_var / len(imgs)
        img_std = torch.sqrt(img_var)
        return img_std.mean()
