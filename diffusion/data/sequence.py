from torch.utils.data import Dataset as _TorchDataset
from torchmanager.data import DataLoader, Dataset
from torchmanager_core import devices, torch
from torchmanager_core.typing import Any, TypeVar, Sequence, Sized, Union

T = TypeVar("T")

class SequenceDataset(Dataset[T]):
    """
    The main unsupervised dataset class to load a PyTorch dataset

    * extends `torchmanager.data.Dataset`

    - Properties:
        - data: A `torch.utils.data.Dataset` or a `Sequence` of data in `T` to load
    """
    data: Union[_TorchDataset[tuple[T, ...]], Sequence[tuple[T, ...]]]

    def __init__(self, data: Union[_TorchDataset[tuple[T, ...]], Sequence[tuple[T, ...]]], batch_size: int, device: torch.device = devices.CPU, drop_last: bool = False, shuffle: bool = False) -> None:
        super().__init__(batch_size, device=device, drop_last=drop_last, shuffle=shuffle)
        self.data = data

    def __getitem__(self, index: Any) -> T:
        x, _ = self.data[index]
        return x

    @property
    def unbatched_len(self) -> int:
        if isinstance(self.data, Sized):
            return len(self.data)
        else:
            dataset = DataLoader(self.data, batch_size=1)
            return len(dataset)
