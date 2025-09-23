from torchmanager.metrics import LPIPS as _LPIPS, LPIPSNetType as LPIPSNet
from torchmanager_core import torch
from torchmanager_core.typing import Protocol, cast
from torchvision import models


class _FeatureExtractor(Protocol):
    @property
    def features(self) -> torch.nn.Sequential:
        ...


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
            case LPIPSNet.VGG16:
                feature_extractor = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            case LPIPSNet.SQUEEZE:
                feature_extractor = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)

        # initialize LPIPS
        feature_extractor = cast(_FeatureExtractor, feature_extractor)
        super().__init__(feature_extractor=feature_extractor.features, net_type=net, target=target)
