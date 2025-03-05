from typing import Collection
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmanager import losses, metrics
from torchmanager.callbacks import Callback
from torchmanager.data import Dataset
from torchmanager_core import devices, torch, view, _raise
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
    __accumulation_steps: int
    __current_batch: int
    scaler: GradScaler | None  # type: ignore

    @property
    def accumulation_steps(self) -> int:
        return self.__accumulation_steps

    @accumulation_steps.setter
    def accumulation_steps(self, s: int) -> None:
        if s < 1:
            raise ValueError("The accumulation steps must be a positive integer.")
        else:
            self.__accumulation_steps = s

    @property
    def current_batch(self) -> int:
        return self.__current_batch

    @current_batch.setter
    def current_batch(self, b: int) -> None:
        if b < 0:
            raise ValueError(f"The batch index must be a non_negative integer, got {b}.")
        else:
            self.__current_batch = b

    @property
    def time_steps(self) -> int:
        return self.raw_model.time_steps

    @time_steps.setter
    def time_steps(self, time_steps: int) -> None:
        self.raw_model.time_steps = time_steps

    @property
    def use_fp16(self) -> bool:
        return self.scaler is not None

    @use_fp16.setter
    def use_fp16(self, use_fp16: bool) -> None:
        if use_fp16 and self.scaler is None:
            assert GradScaler is not NotImplemented, _raise(ImportError("The `torch.cuda.amp` module is not available."))
            self.scaler = GradScaler()  # type: ignore
        elif not use_fp16 and self.scaler is not None:
            self.scaler = None

    def __init__(self, model: DM, optimizer: Optimizer | None = None, loss_fn: losses.Loss | dict[str, losses.Loss] | None = None, metrics: dict[str, metrics.Metric] = {}, use_fp16: bool = False, *, accumulative_steps: int = 1) -> None:
        super().__init__(model, model.time_steps, optimizer, loss_fn, metrics)
        self.accumulation_steps = accumulative_steps

        # initialize fp16 scaler
        if use_fp16:
            assert GradScaler is not NotImplemented, _raise(ImportError("The `torch.cuda.amp` module is not available."))
            self.scaler = GradScaler()  # type: ignore
        else:
            self.scaler = None

    def _train(self, dataset: DataLoader[Any] | Dataset[Any] | Collection[Any], /, iterations: int | None = None, *args, device: torch.device = devices.CPU, use_multi_gpus: bool = False, callbacks_list: list[Callback] = [], **kwargs) -> dict[str, float]:
        self.current_batch = 0
        return super()._train(dataset, iterations, *args, device=device, use_multi_gpus=use_multi_gpus, callbacks_list=callbacks_list, **kwargs)

    def convert(self) -> None:
        if not hasattr(self, 'scaler'):
            self.scaler = None
        super().convert()

    def forward_diffusion(self, data: torch.Tensor, condition: Any = None, t: torch.Tensor | None = None) -> tuple[Any, Any]:
        # initialize
        t = torch.randint(1, self.time_steps + 1, (data.shape[0],), device=data.device).long() if t is None else t.to(data.device)
        return self.raw_model.forward_diffusion(data, t, condition=condition)

    def optimize(self) -> None:
        if self.current_batch % self.accumulation_steps == 0:
            return super().optimize()

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
            view.warnings.warn("The `GradScaler` is only available on CUDA devices. Disabling half precision.")
            self.scaler = None
        return super().to(device)

    def train_step(self, x_train: torch.Tensor | Any, y_train: torch.Tensor | Any, *, forward_diffusion: bool = True) -> dict[str, float]:
        if not self.use_fp16:
            return super().train_step(x_train, y_train, forward_diffusion=forward_diffusion)

        # forward diffusion sampling
        if forward_diffusion:
            assert isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor), "The input and target must be a valid `torch.Tensor`."
            x_t, objective = self.forward_diffusion(y_train.to(x_train.device), condition=x_train)
        else:
            x_t, objective = x_train, y_train

        # forward pass
        with autocast('cuda'):
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
