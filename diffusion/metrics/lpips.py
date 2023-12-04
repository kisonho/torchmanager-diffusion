from lpips import LPIPS as _LPIPS
from torchmanager.metrics import Metric
from torchmanager_core import torch
from torchmanager_core.typing import Any
from typing import Optional


class LPIPS(Metric):
    lpips: _LPIPS

    def __init__(self, target: Optional[str] = None) -> None:
        super().__init__(target=target)
        self.lpips = _LPIPS(net='alex', verbose=False)
        self.lpips.eval()

    @torch.no_grad()
    def forward(self, input: Any, target: Any) -> torch.Tensor:
        gt = target
        img = input
        return self.lpips(gt, img)
