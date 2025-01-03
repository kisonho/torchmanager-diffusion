from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer
from torchmanager import losses, metrics
from torchmanager_core import torch, _raise
from torchmanager_core.typing import Any, TypeVar, cast, overload

from .base import DiffusionManager
from .protocols import DiffusionData, DiffusionModule


DM = TypeVar('DM', bound=DiffusionModule)


class Manager(DiffusionManager[DM]):
    """
    The manager that handles diffusion models

    * extends: `DiffusionManager`
    * Generic: `DM`

    - Properties:
        - scaler: An optional `GradScaler` object to use half precision
        - use_fp16: A `bool` flag to use half precision
    """
    autocaster: autocast | None
    scaler: GradScaler | None

    @property
    def time_steps(self) -> int:
        return self.raw_model.time_steps

    @time_steps.setter
    def time_steps(self, time_steps: int) -> None:
        self.raw_model.time_steps = time_steps

    @property
    def use_fp16(self) -> bool:
        return self.scaler is not None

    def __init__(self, model: DM, optimizer: Optimizer | None = None, loss_fn: losses.Loss | dict[str, losses.Loss] | None = None, metrics: dict[str, metrics.Metric] = {}, use_fp16: bool = False) -> None:
        super().__init__(model, model.time_steps, optimizer, loss_fn, metrics)
        self.scaler = GradScaler("cpu") if use_fp16 else None

        # initialize fp16 scaler
        if use_fp16:
            assert GradScaler is not NotImplemented, _raise(ImportError("The `torch.cuda.amp` module is not available."))
            self.autocaster = autocast('cpu')
            self.scaler = GradScaler()  # type: ignore
        else:
            self.autocaster = self.scaler = None

    def convert(self) -> None:
        if not hasattr(self, 'scaler'):
            self.scaler = None
        super().convert()

    def forward(self, input: Any, target: Any = None) -> tuple[Any, torch.Tensor | None]:
        if self.use_fp16:
            with autocast('cuda'):
                return super().forward(input, target)
        else:
            return super().forward(input, target)

    def forward_diffusion(self, data: torch.Tensor, condition: Any = None, t: torch.Tensor | None = None) -> tuple[Any, Any]:
        # initialize
        t = torch.randint(1, self.time_steps + 1, (data.shape[0],), device=data.device).long() if t is None else t.to(data.device)
        return self.raw_model.forward_diffusion(data, t, condition=condition)

    def optimize(self) -> None:
        if self.use_fp16:
            scaler = cast(GradScaler, self.scaler)
            scaler.step(self.compiled_optimizer)
            scaler.update()
        else:
            super().optimize()

    @overload
    def sampling_step(self, data: DiffusionData, i: int, /) -> torch.Tensor:
        ...

    @overload
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        predicted_noise, _ = self.forward(data)
        return self.raw_model.sampling_step(data, i, predicted_obj=predicted_noise, return_noise=return_noise)

    def to(self, device: torch.device) -> None:
        if device.type != 'cuda' and self.use_fp16:
            self.autocaster = autocast('cpu')
        elif self.use_fp16:
            self.autocaster = autocast('cuda')
        return super().to(device)

    def train_step(self, x_train: Any, y_train: Any, *, forward_diffusion: bool = True) -> dict[str, float]:
        if not self.use_fp16:
            return super().train_step(x_train, y_train, forward_diffusion=forward_diffusion)
        else:
            assert self.autocaster is not None, _raise(RuntimeError("The `autocaster` is not available when using fp16."))

        # forward diffusion sampling
        if forward_diffusion:
            assert isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor), "The input and target must be a valid `torch.Tensor`."
            x_t, objective = self.forward_diffusion(y_train.to(x_train.device), condition=x_train)

        # forward pass
        with self.autocaster:
            y, loss = self.forward(x_t, objective)
        assert loss is not None, _raise(TypeError("Loss cannot be fetched."))

        # backward pass
        assert self.scaler is not None, _raise(RuntimeError("The `GradScaler` is not available."))
        self.compiled_optimizer.zero_grad()
        loss = cast(torch.Tensor, self.scaler.scale(loss))
        self.backward(loss)
        self.scaler.step(self.compiled_optimizer)
        self.scaler.update()
        return self.eval(y, objective)
