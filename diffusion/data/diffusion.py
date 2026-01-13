from torchmanager_core import devices, torch
from dataclasses import dataclass
from typing import Generic, TypeVar


C = TypeVar('C')


@dataclass(frozen=True, slots=True)
class DiffusionData(Generic[C]):
    """
    The data for diffusion model

    * immutable via `frozen=True`
    * memory‑efficient via `slots=True`
    * generic over the condition type `C`

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

    def to(self, device: torch.device):
        """Return a new `DiffusionData` on `device` (dataclasses are immutable here)."""
        condition = devices.move_to_device(self.condition, device)
        return DiffusionData(self.x.to(device), self.t.to(device), condition)

    # Optional: tuple-like conveniences (unpacking and indexing)
    def __iter__(self):
        yield self.x
        yield self.t
        yield self.condition

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int):
        if not isinstance(idx, int):
            raise TypeError("indices must be integers")
        if idx < 0:
            idx += 3
        if idx not in (0, 1, 2):
            raise IndexError("tuple index out of range")
        return (self.x, self.t, self.condition)[idx]
