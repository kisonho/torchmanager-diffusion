from torchmanager_core import torch, view
from torchmanager_core.typing import Type

from .unet import Unet


def build_unet(in_channels: int, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, dim_mults: tuple[int, ...] = (1, 2, 2, 2), dropout: float = 0.1) -> Unet:
    """
    Build the UNET with given input channels

    - Parameters:
        - in_channels: An `int` of input image channels
        - conv_type: A `type` of class that extends `torch.nn.Conv2d` for the convolutional layer type
        - dim_mults: A `tuple` of dimension multiplies in `int`
        - dropout: A `float` of the dropout ratio
    - Returns: A `Unet` which has both its input and output channel of the given `in_channels`
    """
    model = Unet(128, channels=in_channels, conv_type=conv_type, dim_mults=dim_mults, dropout=dropout)
    view.logger.info(model)
    view.logger.info("--------------------------------")
    return model


def build_unet_small(in_channels: int, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, dim_mults: tuple[int, ...] = (1, 2, 4, 8), dropout: float = 0.1) -> Unet:
    """
    Build the UNET implemented in hugging face with given input channels

    - Parameters:
        - in_channels: An `int` of input image channels
        - conv_type: A `type` of class that extends `torch.nn.Conv2d` for the convolutional layer type
        - dim_mults: A `tuple` of dimension multiplies in `int`
        - dropout: A `float` of the dropout ratio
    - Returns: A `Unet` which has both its input and output channel of the given `in_channels`
    """
    model = Unet(32, channels=in_channels, conv_type=conv_type, dim_mults=dim_mults, dropout=dropout)
    view.logger.info(model)
    view.logger.info("--------------------------------")
    return model
