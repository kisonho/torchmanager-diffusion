import torch
from typing import Any, Generic, Iterable, NamedTuple, TypeVar, cast

C = TypeVar('C')


def _move_to_device(target: C, /, device: torch.device, *, recursive: bool = True) -> C:
    if isinstance(target, torch.Tensor):
        moved_target = target.to(device)
    elif isinstance(target, dict):  # if target is a dict
        moved_target = cast(dict[str, Any], target)
        moved_target = {k: _move_to_device(t, device) if recursive else t for k, t in moved_target.items()}
    elif isinstance(target, Iterable):
        moved_target = cast(Iterable[Any], target)
        moved_target = [_move_to_device(t, device) if recursive else t for t in moved_target]
    else:
        moved_target = target
    return cast(C, moved_target)


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
        condition = None if self.condition is None else _move_to_device(self.condition, device)
        return DiffusionData(self.x.to(device), self.t.to(device), condition)


__all__ = ['DiffusionData']
