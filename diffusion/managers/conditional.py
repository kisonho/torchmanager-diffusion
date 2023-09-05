import torch
from typing import Optional, TypeVar

from diffusion.data import DiffusionData
from diffusion.nn import ConditionalDiffusionModule

from .ddpm import DDPMManager

M = TypeVar("M", bound=ConditionalDiffusionModule)


class ConditionalDDPMManager(DDPMManager[M]):
    """Manager for conditional DDPM"""

    def forward_diffusion(self, data: torch.Tensor, condition: Optional[torch.Tensor] = None) -> tuple[DiffusionData, torch.Tensor]:
        x, noise = super().forward_diffusion(data.float())
        x.condition = condition
        return x, noise
