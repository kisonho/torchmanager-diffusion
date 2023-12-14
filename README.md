# Torchmanager Diffusion Models Plug-in
The torchmanager implementation for diffusion models.

## Pre-requisites
* Python >= 3.9
* [PyTorch](https://pytorch.org)
* [torchmanager](https://github.com/kisonho/torchmanager) >= 1.2

## Installation
* PyPi: `pip install --pre torchmanager-diffusion`

## DDPM Manager Usage
### Train DDPM
Direct compile `DDPMManager` to train a DDPM.

```python
import diffusion
from diffusion import DDPMManager
from torchmanager import callbacks, data, losses

# initialize dataset
dataset: data.Dataset = ...

# initialize model, beta_space, and time_steps
model: diffusion.nn.DiffusionModule = ...
beta_space: diffusion.scheduling.BetaSpace = ...
time_steps: int = ...

# initialize optimizer and loss function
optimizer: torch.optim.Optimizer = ...
loss_fn: losses.Loss = ...

# compile the ddpm manager
manager = DDPMManager(model, beta_space, time_steps, optimizer=optimizer, loss_fn=loss_fn)

# initialize callbacks
callback_list: list[callbacks.Callback] = ...

# train the model
trained_model = manager.fit(dataset, epochs=..., callbacks=callback_list)
```

### Evaluate DDPM
Add necessary metrics and use `test` method with `sampling_images` as `True` to evaluate the trained model.

```python
import torch
from diffusion import DDPMManager
from torchmanager import data, metrics
from torchvision import models

# load manager from checkpoints
manager = DDPMManager.from_checkpoint(...)
assert isinstance(manager, DDPMManager), "manager is not a DDPMManager."

# initialize dataset
testing_dataset: data.Dataset = ...

# add neccessary metrics
inception = models.inception_v3(pretrained=True)
inception.fc = torch.nn.Identity()  # type: ignore
inception.eval()
fid = metrics.FID(inception)
manager.metrics.update({"FID": fid})

# evaluate the model
summary = manager.test(testing_dataset, sampling_images=True)
```

## Customize Diffusion Algorithm
Inherit `DiffusionManager` and implement abstract methods `forward_diffusion` and `sampling_step` to customize the diffusion algorithm.

```python
from diffusion import DiffusionManager

class CustomizedManager(DiffusionManager):
    def forward_diffusion(self, data: Any, condition: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> tuple[Any, torch.Tensor]:
        ...

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ...
```
