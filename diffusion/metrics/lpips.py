from lpips import LPIPS as _LPIPS
from torchmanager.metrics import Metric
from torchmanager_core import torch
from torchmanager_core.typing import Any, Enum


class LPIPSNet(Enum):
    """The pre-trained LPIPS network types"""
    ALEX = 'alex'
    SQUEEZE = 'squeeze'
    VGG = 'vgg'


class LPIPS(Metric):
    """
    The LPIPS metric

    - Properties:
        - lpips: The LPIPS module
    """
    lpips: _LPIPS

    def __init__(self, net: LPIPSNet = LPIPSNet.ALEX, target: str | None = None) -> None:
        super().__init__(target=target)
        self.lpips = _LPIPS(net=net.value, verbose=False)  # type: ignore
        self.lpips.eval()

    @torch.no_grad()
    def forward(self, input: Any, target: Any) -> torch.Tensor:
        lpips: torch.Tensor = self.lpips(target, input)
        return lpips.squeeze().mean()
