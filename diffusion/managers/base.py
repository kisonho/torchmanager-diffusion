from torch.nn.utils import clip_grad
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmanager import losses, metrics, Manager as _Manager
from torchmanager.data import Dataset
from torchmanager_core import abc, devices, errors, torch, view, _raise
from torchmanager_core.typing import Any, Module, Sequence, overload

from .protocols import DiffusionData


class DiffusionManager(_Manager[Module], abc.ABC):
    """
    The basic `Manager` for diffusion models

    * Abstract class
    * Extends: `torchmanager.Manager`

    - Properties:
        - time_steps: An `int` of total time steps
    - Methods to implement:
        - forward_diffusion: Forward pass of diffusion model, sample noises
        - sampling_step: Sampling step of diffusion model
    """
    time_steps: int

    def __init__(self, model: Module, time_steps: int, optimizer: Optimizer | None = None, loss_fn: losses.Loss | dict[str, losses.Loss] | None = None, metrics: dict[str, metrics.Metric] = {}) -> None:
        """
        Constructor

        - Prarameters:
            - model: An optional target `torch.nn.Module` to be trained
            - time_steps: An `int` of total number of steps
            - optimizer: An optional `torch.optim.Optimizer` to train the model
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
        """
        # initialize
        super().__init__(model, optimizer, loss_fn, metrics)
        self.time_steps = time_steps

    def backward(self, loss: torch.Tensor) -> None:
        super().backward(loss)
        clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=1)

    @abc.abstractmethod
    def forward_diffusion(self, data: Any, condition: Any = None, t: torch.Tensor | None = None) -> tuple[Any, Any]:
        """
        Forward pass of diffusion model, sample noises

        - Parameters:
            - data: Any kind of noised data
            - condition: An optional `Any` kind of the condition to generate images
            - t: An optional `torch.Tensor` of the time step, sampling uniformly if not given
        - Returns: A `tuple` of `Any` kind of wrapped noisy images and sampled time step and `Any` kind of training objective
        """
        return NotImplemented

    @torch.no_grad()
    def predict(self, num_images: int, image_size: int | tuple[int, ...], *args: Any, condition: torch.Tensor | None = None, noises: torch.Tensor | None = None, sampling_range: Sequence[int] | range | None = None, device: torch.device | list[torch.device] | None = None, empty_cache: bool = True, use_multi_gpus: bool = False, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        # find available device
        cpu, device, target_devices = devices.search(device)
        if device == cpu and len(target_devices) < 2:
            use_multi_gpus = False
        devices.set_default(target_devices[0])

        # initialize and format parameters
        image_size = image_size if isinstance(image_size, tuple) else (3, image_size, image_size)
        assert image_size[0] > 0 and image_size[1] > 0, _raise(ValueError(f"Image size must be positive numbers, got {image_size}."))
        assert num_images > 0, _raise(ValueError(f"Number of images must be a positive number, got {num_images}."))
        imgs = torch.randn([num_images] + list(image_size)) if noises is None else noises
        assert imgs.shape[0] >= num_images, _raise(ValueError(f"Number of noises ({imgs.shape[0]}) must be equal or greater than number of images to generate ({num_images})"))

        try:
            # move model to device
            if use_multi_gpus:
                self.data_parallel(target_devices)
            else:
                imgs = imgs.to(device)
            self.to(device)
            self.model.eval()

            # move condition to device
            c = devices.move_to_device(condition, device) if condition is not None else None
            if c is not None:
                assert isinstance(c, torch.Tensor), "Condition must be a valid `torch.Tensor` when given."
            return self.sampling(num_images, imgs, *args, condition=c, sampling_range=sampling_range, show_verbose=show_verbose, **kwargs)
        except Exception as error:
            view.logger.error(error)
            runtime_error = errors.PredictionError()
            raise runtime_error from error
        finally:
            # empty cache
            if empty_cache:
                self.to(cpu)
                self.model = self.raw_model
                self.loss_fn = self.raw_loss_fn if self.raw_loss_fn is not None else self.raw_loss_fn
                devices.empty_cache()
    @overload
    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> torch.Tensor:
        ...

    @overload
    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abc.abstractmethod
    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Sampling step of diffusion model

        - Parameters:
            - data: A `DiffusionData` object
            - i: An `int` of current time step
            - return_noise: A `bool` flag to return predicted noise
        - Returns: A `torch.Tensor` of noised image if not returning noise or a `tuple` of noised image and predicted noise in `torch.Tensor` if returning noise
        """
        return NotImplemented

    @torch.no_grad()
    def sampling(self, num_images: int, x_t: torch.Tensor, *args: Any, condition: torch.Tensor | None = None, sampling_range: Sequence[int] | range | None = None, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        '''
        Samples a given number of images

        - Parameters:
            - num_images: An `int` of number of images to generate
            - x_t: A `torch.Tensor` of the image at T step
            - condition: An optional `torch.Tensor` of the condition to generate images
            - sampling_range: An optional `Sequence[int]`, or `range` of the range of time steps to sample
            - start_index: An optional `int` of the start index of the time step
            - end_index: An `int` of the end index of the time step
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Retruns: A `list` of `torch.Tensor` generated results
        '''
        # initialize
        imgs = x_t
        sampling_range = range(self.time_steps, 0, -1) if sampling_range is None else sampling_range
        progress_bar = view.tqdm(desc='Sampling loop time step', total=len(sampling_range), disable=not show_verbose)

        # sampling loop time step
        try:
            for i, t in enumerate(sampling_range):
                # fetch data
                t = torch.full((num_images,), t, dtype=torch.long, device=imgs.device)

                # append to predicitions
                x = DiffusionData(imgs, t, condition=condition)
                y = self.sampling_step(x, len(sampling_range) - i)
                imgs = y.to(imgs.device)
                progress_bar.update()

            # reset model and loss
            return [img for img in imgs]
        finally:
            progress_bar.close()

    @torch.no_grad()
    def test(self, dataset: DataLoader[torch.Tensor] | Dataset[torch.Tensor], *args: Any, sampling_images: bool = False, sampling_shape: int | tuple[int, ...] | None = None, sampling_range: Sequence[int] | range | None = None, device: torch.device | list[torch.device] | None = None, empty_cache: bool = True, use_multi_gpus: bool = False, show_verbose: bool = False, **kwargs: Any) -> dict[str, float]:
        """
        Test target model

        - Parameters:
            - dataset: A `torch.utils.data.DataLoader` or `.data.Dataset` dataset
            - *args: An optional `tuple` of `Any` of additional arguments for sampling
            - sampling_images: A `bool` flag to sample images during testing
            - sampling_shape: An optional `int` or `tuple` of `int` of the shape of sampled images
            - sampling_range: An optional `Sequence[int]`, or `range` of the range of time steps to sample
            - device: An optional `torch.device` to test on if not using multi-GPUs or an optional default `torch.device` for testing otherwise
            - empyt_cache: A `bool` flag to empty cache after testing
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
            - show_verbose: A `bool` flag to show the progress bar during testing
            - **kwargs: An optional `dict` of `Any` of additional keyword arguments for sampling
        - Returns: A `dict` of validation summary
        """
        # normali testing if not sampling images
        if not sampling_images:
            return super().test(dataset, device=device, empty_cache=empty_cache, use_multi_gpus=use_multi_gpus, show_verbose=show_verbose)

        # initialize device
        cpu, device, target_devices = devices.search(device)
        if device == cpu and len(target_devices) < 2:
            use_multi_gpus = False
        devices.set_default(target_devices[0])

        # initialize
        summary: dict[str, float] = {}
        batched_len = dataset.batched_len if isinstance(dataset, Dataset) else len(dataset)

        # reset loss function and metrics
        for _, m in self.metric_fns.items():
            m.eval().reset()

        try:
            # set module status and move to device
            if use_multi_gpus:
                self.data_parallel(target_devices)
            self.to(device)
            self.model.eval()

            # batch loop
            for b, (x_test, y_test) in enumerate(dataset):
                # move x_test, y_test to device
                if not use_multi_gpus:
                    x_test = devices.move_to_device(x_test, device)
                y_test = devices.move_to_device(y_test, device)
                assert isinstance(x_test, torch.Tensor), "The input must be a valid `torch.Tensor`."
                assert isinstance(y_test, torch.Tensor), "The target must be a valid `torch.Tensor`."

                # sampling
                view.logger.info(f"Sampling images {b + 1}/{batched_len}...")
                sampling_shape = y_test.shape[-3:] if sampling_shape is None else sampling_shape
                noises = torch.randn_like(y_test, dtype=torch.float, device=y_test.device)
                x = self.sampling(int(x_test.shape[0]), noises, *args, condition=x_test, sampling_range=sampling_range, show_verbose=show_verbose, **kwargs)
                x = torch.cat([img.unsqueeze(0) for img in x])
                x = devices.move_to_device(x, y_test.device)
                step_summary = self.eval(x, y_test)

                # initialize summary info
                summary_info: str = f"Step {b + 1}/{batched_len}: "

                # add metrics to summary
                for i, (name, value) in enumerate(step_summary.items()):
                    summary_info += ", " if i > 0 else ""
                    summary_info += f"{name}={value:.4f}"

                # log summary info
                view.logger.info(summary_info)

            # summarize
            for name, fn in self.metric_fns.items():
                if name.startswith("val_"):
                    name = name.replace("val_", "")
                try:
                    summary[name] = float(fn.result.detach())
                except Exception as metric_error:
                    runtime_error = errors.MetricError(name)
                    raise runtime_error from metric_error

            # reset model and loss
            return summary
        except KeyboardInterrupt:
            view.logger.info("Testing interrupted.")
            return {}
        except Exception as error:
            view.logger.error(error)
            runtime_error = errors.TestingError()
            raise runtime_error from error
        finally:
            # empty cache
            if empty_cache:
                self.to(cpu)
                self.model = self.raw_model
                self.loss_fn = self.raw_loss_fn if self.raw_loss_fn is not None else self.raw_loss_fn
                devices.empty_cache()

    def to(self, device: torch.device) -> None:
        super().to(device)

    def train_step(self, x_train: Any, y_train: Any, *, forward_diffusion: bool = True) -> dict[str, float]:
        # forward diffusion sampling
        if forward_diffusion:
            assert isinstance(x_train, torch.Tensor) and isinstance(y_train, torch.Tensor), "The input and target must be a valid `torch.Tensor`."
            x_train_noise, objective = self.forward_diffusion(y_train.to(x_train.device), condition=x_train)
        else:
            x_train_noise, objective = x_train, y_train
        return super().train_step(x_train_noise, objective)

    def test_step(self, x_test: Any, y_test: Any, *, forward_diffusion: bool = True) -> dict[str, float]:
        # forward diffusion sampling
        if forward_diffusion:
            assert isinstance(x_test, torch.Tensor) and isinstance(y_test, torch.Tensor), "The input and target must be a valid `torch.Tensor`."
            x_test_noise, objective = self.forward_diffusion(y_test.to(x_test.device), condition=x_test)
        else:
            x_test_noise, objective = x_test, y_test
        return super().test_step(x_test_noise, objective)
