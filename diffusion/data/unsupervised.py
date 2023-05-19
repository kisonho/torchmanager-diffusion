from torch.utils.data import Dataset as _TorchDataset
from torchmanager.data import DataLoader, Dataset as _Dataset
from torchmanager_core import abc, devices, torch
from torchmanager_core.typing import Any, TypeVar, Sequence, Sized, Union

T = TypeVar("T")


class UnsupervisedDataset(_Dataset[T], abc.ABC):
    """
    A specific dataset for unsupervised learning

    * extends: `torchmanager.data.Dataset`
    * Abstract class

    - Methods to implement:
        - unbatched_len: A property method that returns the total length of unbatched dataset
        - __getitem__: The built in method to get items by index (as in `torch.utils.data.Dataset`)
    """
    @staticmethod
    def unpack_data(data: Any) -> T:
        """
        Unpacks a single data into inputs and targets

        - Parameters:
            - data: `Any` kind of single data
        - Returns: `Any` kind of inputs with type `T`
        """
        if isinstance(data, Sequence):
            return data[0] if len(data) >= 2 else NotImplemented  # type: ignore
        else:
            return data


class Dataset(UnsupervisedDataset[T]):
    """
    The main unsupervised dataset class to load a PyTorch dataset

    * extends `.unsupervised.UnsupervisedDataset`

    - Properties:
        - data: A `torch.utils.data.Dataset` or a `Sequence` of data in `T` to load
    """
    data: Union[_TorchDataset[T], Sequence[T]]

    def __init__(self, data: Union[_TorchDataset[T], Sequence[T]], batch_size: int, device: torch.device = devices.CPU, drop_last: bool = False, shuffle: bool = False) -> None:
        super().__init__(batch_size, device=device, drop_last=drop_last, shuffle=shuffle)
        self.data = data

    def __getitem__(self, index: Any) -> T:
        return self.data[index]

    @property
    def unbatched_len(self) -> int:
        if isinstance(self.data, Sized):
            return len(self.data)
        else:
            dataset = DataLoader(self.data, batch_size=1)
            return len(dataset)
