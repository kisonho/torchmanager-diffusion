from lpips import LPIPS as _LPIPS
from torchmanager.metrics import Metric
from torchmanager_core import torch
from torchmanager_core.typing import Any, Enum
from typing import Optional


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

    def __init__(self, net: LPIPSNet = LPIPSNet.ALEX, target: Optional[str] = None) -> None:
        super().__init__(target=target)
        self.lpips = _LPIPS(net=net.value, verbose=False)
        self.lpips.eval()

    @torch.no_grad()
    def forward(self, input: Any, target: Any) -> torch.Tensor:
        gt = target
        img = input
        lpips: torch.Tensor = self.lpips(gt, img)
        return lpips.squeeze().mean()
