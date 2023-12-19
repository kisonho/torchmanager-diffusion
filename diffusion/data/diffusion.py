from torchmanager_core import torch
from torchmanager_core.typing import Optional, Self


class DiffusionData:
    """
    The data for diffusion model

    * implements: `nn.protocols.TimedData`, `torchmanager_core.devices.DeviceMovable`

    - Properties:
        - x: A `torch.Tensor` of the main data
        - t: A `torch.Tensor` of the time
        - condition: An optional `torch.Tensor` of the condition data
    """
    x: torch.Tensor
    """A `torch.Tensor` of the main data"""
    t: torch.Tensor
    """A `torch.Tensor` of the time"""
    condition: Optional[torch.Tensor]

    def __init__(self, x: torch.Tensor, t: torch.Tensor, /, condition: Optional[torch.Tensor] = None) -> None:
        self.x = x
        self.t = t
        self.condition = condition

    def to(self, device: torch.device) -> Self:
        self.x = self.x.to(device)
        self.t = self.t.to(device)
        if self.condition is not None:
            self.condition = self.condition.to(device)
        return self
