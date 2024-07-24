import torch
from functools import partial
from typing import Optional, Type

from diffusion.nn import Attention, ConvNextBlock, LinearAttention, PreNorm, Residual, ResnetBlock, SinusoidalPositionEmbeddings, TimedModule
from diffusion.nn.diffusion.protocols import TimedData


class UNet(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            /,
            init_dim: Optional[int] = None,
            out_dim: Optional[int] = None,
            dropout: float = 0,
            dim_mults: tuple[int, ...] = (1, 2, 4, 8),
            channels: int = 3,
            conv_type: Type[torch.nn.Conv2d] = torch.nn.Conv2d,
            with_time_emb: bool = True,
            resnet_block_groups: int = 8,
            use_convnext: bool = False,
            convnext_mult: int = 2,
    ) -> None:
        super().__init__()
        Conv2d = conv_type

        # determine dimensions
        self.channels = channels
        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = Conv2d(channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        if use_convnext:
            block_type = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_type = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = torch.nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                torch.nn.Linear(dim, time_dim),
                torch.nn.GELU(),
                torch.nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        block_type(dim_in, dim_out, conv_type=Conv2d, dropout=dropout, time_emb_dim=time_dim),
                        block_type(dim_out, dim_out, conv_type=Conv2d, dropout=dropout, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Conv2d(dim_out, dim_out, 3, 2, 1) if not is_last else torch.nn.Identity(),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = block_type(mid_dim, mid_dim, conv_type=Conv2d, dropout=dropout, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_type(mid_dim, mid_dim, conv_type=Conv2d, dropout=dropout, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        block_type(dim_out * 2, dim_in, conv_type=Conv2d, dropout=dropout, time_emb_dim=time_dim),
                        block_type(dim_in, dim_in, conv_type=Conv2d, dropout=dropout, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        torch.nn.ConvTranspose2d(dim_in, dim_in, 3, 2, 1, output_padding=1) if not is_last else torch.nn.Identity(),
                    ]
                )
            )
        out_dim = out_dim if out_dim is not None else channels
        self.final_conv = torch.nn.Sequential(
            block_type(dim, dim, conv_type=Conv2d), Conv2d(dim, out_dim, 1)
        )

    def forward(self, x: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.init_conv(x)
        t = self.time_mlp(time) if self.time_mlp is not None else None
        h = []

        # downsample
        for down in self.downs:
            assert isinstance(down, torch.nn.ModuleList), "Down module is not a valid module list"
            block1, block2, attn, downsample = down
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for up in self.ups:
            assert isinstance(up, torch.nn.ModuleList), "Up module is not a valid module list"
            block1, block2, attn, upsample = up
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)


class TimedUNet(TimedModule, UNet):
    def unpack_data(self, x_in: TimedData) -> tuple[torch.Tensor, ...]:
        return x_in.x, x_in.t


Unet = TimedUNet
