from torchmanager_core import devices, torch
from torchmanager_core.typing import Generic, NamedTuple, TypeVar


C = TypeVar('C')


class DiffusionData(NamedTuple, Generic[C]):
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
    condition: C | None = None
    """An optional `C` of the condition data"""

    def to(self, device: torch.device):
        condition = devices.move_to_device(self.condition, device)
        return DiffusionData(self.x.to(device), self.t.to(device), condition)
