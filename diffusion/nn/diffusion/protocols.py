import torch
from typing import Optional, Protocol


class TimedData(Protocol):
    @property
    def x(self) -> torch.Tensor:
        return NotImplemented

    @property
    def t(self) -> torch.Tensor:
        return NotImplemented

    @property
    def condition(self) -> Optional[torch.Tensor]:
        return NotImplemented
