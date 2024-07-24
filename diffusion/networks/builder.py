from torchmanager_core import torch, view
from torchmanager_core.typing import Type, overload

from .unet import TimedUNet, UNet


@overload
def build(in_channels: int, out_channels: int, /, *, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, dim_mults: tuple[int, ...] = (1, 2, 2, 2), dropout: float = 0.1, with_time_emb: bool = True) -> UNet: ...


@overload
def build(in_channels: int, out_channels: int, /, *, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, dim_mults: tuple[int, ...] = (1, 2, 2, 2), dropout: float = 0.1, use_timed_data: bool = True, with_time_emb: bool = True) -> TimedUNet: ...


def build(in_channels: int, out_channels: int, /, *, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, dim_mults: tuple[int, ...] = (1, 2, 2, 2), dropout: float = 0.1, use_timed_data: bool = False, with_time_emb: bool = True) -> UNet:
    """
    Build the UNET with given input channels. This is the UNET same as the one implemented with TensorFlow in DDPM paper.

    - Parameters:
        - in_channels: An `int` of input image channels
        - conv_type: A `type` of class that extends `torch.nn.Conv2d` for the convolutional layer type
        - dim_mults: A `tuple` of dimension multiplies in `int`
        - dropout: A `float` of the dropout ratio
    - Returns: A `TimedUNet` or `Unet` which has both its input and output channel of the given `in_channels`
    """
    model = TimedUNet(128, channels=in_channels, out_dim=out_channels, conv_type=conv_type, dim_mults=dim_mults, dropout=dropout, with_time_emb=with_time_emb) if use_timed_data else UNet(128, channels=in_channels, out_dim=out_channels, conv_type=conv_type, dim_mults=dim_mults, dropout=dropout, with_time_emb=with_time_emb)
    view.logger.info(model)
    view.logger.info("--------------------------------")
    return model


def build_unet(in_channels: int, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, dim_mults: tuple[int, ...] = (1, 2, 2, 2), dropout: float = 0.1) -> TimedUNet:
    """
    Build the UNET with given input channels. This is the UNET same as the one implemented with TensorFlow in DDPM paper.

    - Parameters:
        - in_channels: An `int` of input image channels
        - conv_type: A `type` of class that extends `torch.nn.Conv2d` for the convolutional layer type
        - dim_mults: A `tuple` of dimension multiplies in `int`
        - dropout: A `float` of the dropout ratio
    - Returns: A `Unet` which has both its input and output channel of the given `in_channels`
    """
    return build(in_channels, in_channels, conv_type=conv_type, dim_mults=dim_mults, dropout=dropout, use_timed_data=True)


def build_unet_small(in_channels: int, conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d, dim_mults: tuple[int, ...] = (1, 2, 4, 8), dropout: float = 0.1) -> UNet:
    """
    Build the UNET with given input channels. This is the same as the one implemented in hugging face.

    - Parameters:
        - in_channels: An `int` of input image channels
        - conv_type: A `type` of class that extends `torch.nn.Conv2d` for the convolutional layer type
        - dim_mults: A `tuple` of dimension multiplies in `int`
        - dropout: A `float` of the dropout ratio
    - Returns: A `Unet` which has both its input and output channel of the given `in_channels`
    """
    model = UNet(32, channels=in_channels, conv_type=conv_type, dim_mults=dim_mults, dropout=dropout)
    view.logger.info(model)
    view.logger.info("--------------------------------")
    return model
