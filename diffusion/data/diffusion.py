from typing import Generic, NamedTuple, TypeVar

from torchmanager_core import devices, torch

C = TypeVar('C')


class DiffusionData(NamedTuple, Generic[C]):
    """
    The data for diffusion model.

    Implemented as a named tuple so `torch.nn.DataParallel` can scatter and move
    the contained tensors across CUDA devices during multi-GPU execution.

    - Properties:
        - x: A `torch.Tensor` of the main data
        - t: A `torch.Tensor` of the time
        - condition: An optional `C` of the condition data
    """

    x: torch.Tensor
    """A `torch.Tensor` of the main data"""

    t: torch.Tensor
    """A `torch.Tensor` of the time"""

    condition: C | None = None
    """An optional `C` of the condition data"""

    def to(self, device: torch.device) -> "DiffusionData[C]":
        condition = devices.move_to_device(self.condition, device)
        return DiffusionData(self.x.to(device), self.t.to(device), condition)
