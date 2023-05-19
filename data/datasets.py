from torchmanager_core import devices, torch
from torchmanager_core.typing import Callable, Enum, Optional, Union, TypeVar
from torchvision import datasets, transforms

from diffusion.data import Dataset

T = TypeVar("T")


class Datasets(Enum):
    """Supported datasets"""
    CIFAR10 = "cifar10"
    MNIST = "mnist"

    def load(self, root_dir: str, batch_size: int, device: torch.device = devices.GPU, image_size: Optional[Union[int, tuple[int, int]]] = None) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor], int, Union[int, tuple[int, int]]]:
        """
        Load the dataset

        - Parameters:
            - root_dir: A `str` of dataset root directory
            - batch_size: An `int` of the batch size
            - image_size: An optional of image size in `int` or a `tuple` of two `int` for the dimensions
        - Returns: A `tuple` of a training dataset in `Dataset` contains `torch.Tensor` elements, a testing dataset in `Dataset` contains `torch.Tensor` elements, an `int` of input channels, and an `int` of image size or `tuple` of image size dimensions in `int`
        """
        if self == Datasets.CIFAR10:
            return load_cifar10(root_dir, batch_size, device=device, image_size=image_size)
        elif self == Datasets.MNIST:
            return load_mnist(root_dir, batch_size, device=device, image_size=image_size)
        else:
            raise NotImplementedError(f"Loader for dataset {self} has not been implemented.")


def load_cifar10(root_dir: str, batch_size: int, device: torch.device = devices.GPU, image_size: Optional[Union[int, tuple[int, int]]] = None, normalize: Optional[transforms.Normalize] = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]), random_flip: bool = True) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor], int, Union[int, tuple[int, int]]]:
    """
    Load the cifar 10 dataset

    - Parameters:
        - root_dir: A `str` of dataset root directory
        - batch_size: An `int` of the batch size
        - image_size: An optional of image size in `int` or a `tuple` of two `int` for the dimensions
        - normalize: An optional normalization preprocess transform in `torchvision.transforms.Normalize`
        - random_flip: A `bool` flag of if random flipping images for training dataset
    - Returns: A `tuple` of a training dataset in `Dataset` contains `torch.Tensor` elements, a testing dataset in `Dataset` contains `torch.Tensor` elements, an `int` of input channels, and an `int` of image size or `tuple` of image size dimensions in `int`
    """
    # initialize preprocessing
    training_transforms: list[Callable[[torch.Tensor], torch.Tensor]] = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ] if random_flip else [transforms.ToTensor()]
    testing_transforms: list[Callable[[torch.Tensor], torch.Tensor]] = [
        transforms.ToTensor(),
    ]

    # initialize image size
    if image_size is None:
        image_size = (32, 32)
    else:
        resize = transforms.Resize(image_size)
        training_transforms.append(resize)
        testing_transforms.append(resize)

    # add normalization
    if normalize is not None:
        training_transforms.append(normalize)
        testing_transforms.append(normalize)

    # wrap into compose
    training_preprocess = transforms.Compose(training_transforms)
    testing_preprocess = transforms.Compose(testing_transforms)

    # load dataset
    cifar10_training = datasets.CIFAR10(root_dir, transform=training_preprocess, download=True)
    training_dataset: Dataset[torch.Tensor] = Dataset(cifar10_training, batch_size, device=device, drop_last=True, shuffle=True)
    cifar10_testing = datasets.CIFAR10(root_dir, train=False, transform=testing_preprocess, download=True)
    testing_dataset: Dataset[torch.Tensor] = Dataset(cifar10_testing, batch_size, device=device)
    return training_dataset, testing_dataset, 3, image_size

def load_mnist(root_dir: str, batch_size: int, device: torch.device = devices.GPU, image_size: Optional[Union[int, tuple[int, int]]] = None) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor], int, Union[int, tuple[int, int]]]:
    # initialize image size
    if image_size is None:
        image_size = (28, 28)
    elif not isinstance(image_size, tuple):
        image_size = (image_size, image_size)

    # initialize preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load dataset
    mnist_training = datasets.MNIST(root_dir, transform=preprocess, download=True)
    training_dataset: Dataset[torch.Tensor] = Dataset(mnist_training, batch_size, device=device, drop_last=True, shuffle=True)
    mnist_testing = datasets.MNIST(root_dir, train=False, transform=preprocess, download=True)
    testing_dataset: Dataset[torch.Tensor] = Dataset(mnist_testing, batch_size, device=device)
    return training_dataset, testing_dataset, 1, image_size