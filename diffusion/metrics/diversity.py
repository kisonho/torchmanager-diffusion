from torchmanager.metrics import Metric
from torchmanager_core import torch
from torchmanager_core.typing import Any
from typing import Optional

from bbdm_new.utils import img_denormalization


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
        imgs = img_denormalization(input) * 255
        return imgs.std(dim=self.sample_dim).mean()
