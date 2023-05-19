from torchmanager_core import torch
from torchmanager_core.typing import Protocol


class TimedData(Protocol):
    x: torch.Tensor
    t: torch.Tensor
