from torchmanager_core import torch
from torchmanager_core.typing import Optional, Protocol


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
