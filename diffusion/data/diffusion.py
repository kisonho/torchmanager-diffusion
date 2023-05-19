from torchmanager_core import torch
from torchmanager_core.typing import Self


class DiffusionData:
    """
    The data for diffusion model

    * implements: `networks.protocols.TimedData`, `torchmanager_core.devices.DeviceMovable`

    - Properties:
        - x: A `torch.Tensor` of the main data
        - t: A `torch.Tensor` of the time
    """
    x: torch.Tensor
    """A `torch.Tensor` of the main data"""
    t: torch.Tensor
    """A `torch.Tensor` of the time"""

    def __init__(self, x: torch.Tensor, t: torch.Tensor) -> None:
        self.x = x
        self.t = t

    def to(self, device: torch.device) -> Self:
        self.x = self.x.to(device)
        self.t = self.t.to(device)
        return self
