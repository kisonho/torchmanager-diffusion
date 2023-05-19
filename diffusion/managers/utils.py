from torchmanager_core import torch
from torchmanager_core.typing import Collection


def _get_index_from_list(vals: torch.Tensor, t: torch.Tensor, x_shape: Collection[int]) -> torch.Tensor:
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    vals = vals.gather(-1, t)
    return vals.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
