import copy, torch
from torch.optim.optimizer import Optimizer
from typing import Any, Callable, Generic, Optional, TypeVar

M = TypeVar('M', bound=torch.nn.Module)
O = TypeVar('O', bound=Optimizer)


class EMAOptimizer(Optimizer, Generic[O, M]):
    """
    An optimizer wrapper that maintains Exponential Moving Average (EMA) of model parameters.

    - Properties:
        - optimizer: The base optimizer to wrap.
        - ema_decay: The decay factor for EMA.
        - model: The model whose parameters will be used for EMA.
        - ema_model: The model whose parameters will be used for EMA.
    """
    __ema_decay: float
    base_optimizer: O
    is_ema_parameters: bool

    @property
    def ema_decay(self) -> float:
        return self.__ema_decay

    @ema_decay.setter
    def ema_decay(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError("EMA decay should be in the range (0, 1).")
        self.__ema_decay = value

    def __init__(self, optimizer: O, model: M, *, defaults: dict[str, Any] = {}, ema_decay: float = 0.999) -> None:
        """
        Wrap a base optimizer and maintain Exponential Moving Average (EMA) of model parameters.

        - Parameters:
            - optimizer: The base optimizer to wrap in `torch.optim.optimizer.Optimizer`.
            - model: The model whose parameters will be used for EMA.
            - ema_decay: The decay factor for EMA in `float`.
        """
        self.base_optimizer = optimizer
        self.ema_decay = ema_decay
        self.is_ema_parameters = False
        self.model = model

        # Create a deep copy of model parameters for EMA
        self.ema_model = copy.deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad = False  # EMA parameters should not be trainable
        super().__init__(self.ema_model.parameters(), optimizer.defaults)

    def __enter__(self) -> None:
        """
        Swap the model parameters with EMA parameters when entering the context.
        """
        if not self.is_ema_parameters:
            self.swap_parameters()
        self.is_ema_parameters = True

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Restore the original model parameters when exiting the context.
        """
        if self.is_ema_parameters:
            self.swap_parameters()
        self.is_ema_parameters = False

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step and update EMA parameters.

        - Parameters
            - closure: An optional closure function that returns the loss value in `float`.
        Returns: An optional loss value in `float` after the optimizer step, if available.
        """
        # Step with the base optimizer
        loss = self.base_optimizer.step(closure)

        # Update EMA weights
        self.update_ema_parameters()
        return loss

    def update_ema_parameters(self) -> None:
        """
        Update the EMA parameters using the current model parameters.
        """
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=(1.0 - self.ema_decay))

    def state_dict(self) -> dict[str, Any]:
        """
        Save the state of both the base optimizer and the EMA parameters.

        Returns: A `dict` containing the state of the optimizer and EMA model with keys in `str`.
        """
        return {
            'optimizer': self.base_optimizer.state_dict(),
            'ema_model': self.ema_model.state_dict()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state for both the base optimizer and the EMA parameters.

        - Parameters:
            - state_dict: The state `dict` containing both optimizer and EMA states with keys in `str`.
        """
        self.base_optimizer.load_state_dict(state_dict['optimizer'])
        self.ema_model.load_state_dict(state_dict['ema_model'])

    def swap_parameters(self) -> None:
        """
        Swap the model's parameters with the EMA parameters for evaluation.
        """
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            param.data, ema_param.data = ema_param.data, param.data
        self.is_ema_parameters = not self.is_ema_parameters

    def use_ema_parameters(self) -> 'EMAOptimizer[O, M]':
        """
        Temporarily replace the model parameters with the EMA parameters.
        Use this during evaluation.
        """
        return self

    def zero_grad(self) -> None:
        """
        Zero the gradients of the base optimizer.
        """
        self.base_optimizer.zero_grad()
