import torch
from typing import Any, Optional, Protocol, Sequence, Union


class Samplable(Protocol):
    """A protocol for samplable objects."""
    def predict(self, num_images: int, image_size: Union[int, tuple[int, ...]], *args: Any, condition: Optional[torch.Tensor] = None, noises: Optional[torch.Tensor] = None, sampling_range: Optional[Union[Sequence[int], range]] = None, device: Optional[Union[torch.device, list[torch.device]]] = None, empty_cache: bool = True, use_multi_gpus: bool = False, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        """Sample images from optional given conditions."""
        ...
