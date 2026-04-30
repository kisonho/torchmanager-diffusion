from torchmanager.metrics import LPIPS as _LPIPS, LPIPSNetType as LPIPSNet
from torchmanager_core import torch
from torchmanager_core.typing import Any, Protocol, cast
from torchvision import models


class _FeatureExtractor(Protocol):
    @property
    def features(self) -> torch.nn.Sequential:
        ...


def _load_lin_layers(checkpoint_name: str, channels: list[int]) -> torch.nn.ModuleList:
    state_dict = torch.hub.load_state_dict_from_url(
        f"https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/{checkpoint_name}.pth",
        file_name=f"lpips_{checkpoint_name}.pth",
        map_location="cpu",
        progress=False,
    )

    lin_layers = torch.nn.ModuleList([
        torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Conv2d(channel, 1, kernel_size=1, bias=False),
        )
        for channel in channels
    ])
    lin_layers.load_state_dict({
        f"{i}.1.weight": state_dict[f"lin{i}.model.1.weight"]
        for i in range(len(channels))
    })
    return lin_layers


class LPIPS(_LPIPS):
    """
    The wrapped LPIPS metric

    - Properties:
        - lpips: The LPIPS module to extract features
    """
    def __init__(self, net: LPIPSNet = LPIPSNet.ALEX, target: str | None = None) -> None:
        """
        Constructor

        - Parameters:
            - net: The `LPIPSNet` to extract features
            - target: A `str` of target name in `input` and `target` during direct calling
        """
        # load pretrained models
        match net:
            case LPIPSNet.ALEX:
                feature_extractor = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
                checkpoint_name = "alex"
                channels = [64, 192, 384, 256, 256]
            case LPIPSNet.VGG16:
                feature_extractor = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                checkpoint_name = "vgg"
                channels = [64, 128, 256, 512, 512]
            case LPIPSNet.SQUEEZE:
                feature_extractor = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
                checkpoint_name = "squeeze"
                channels = [64, 128, 256, 384, 384, 512, 512]
        lin_layers = _load_lin_layers(checkpoint_name, channels)

        # initialize LPIPS
        feature_extractor = cast(_FeatureExtractor, feature_extractor)
        super().__init__(feature_extractor=feature_extractor.features, net_type=net, lin_layers=lin_layers, target=target)

    def forward_features(self, x: torch.Tensor) -> Any:
        # check channels
        c = x.shape[1]

        # expand if only one channel detected for the input
        if c == 1:
            x = x.expand((-1, c * 3, -1, -1))
        return super().forward_features(x)
    
__all__ = ['LPIPS']
