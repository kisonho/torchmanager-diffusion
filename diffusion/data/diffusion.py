from torchmanager_core import torch
from torchmanager_core.typing import NamedTuple, Optional


class DiffusionData(NamedTuple):
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
    condition: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        return DiffusionData(self.x.to(device), self.t.to(device), self.condition.to(device) if self.condition is not None else None)
