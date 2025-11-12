import copy, torch
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer
from typing import Any, Callable, Iterable, Generic, TypedDict, TypeVar, cast

O = TypeVar('O', bound=Optimizer)


class EMAState(TypedDict):
    """
    A dictionary to store the state of the EMA optimizer.

    - Properties:
        - ema_params: The parameters used for EMA.
        - optimizer: The state of the base optimizer.
        - params: The original parameters tracked.
        - ema_decay: The decay factor for EMA.
    """
    optim_state: dict[str, Any]
    ema_decay: float
    ema_params: Iterable[Parameter]
    base_optimizer: Optimizer
    params: Iterable[Parameter]


class EMAOptimizer(Optimizer, Generic[O]):
    """
    An optimizer wrapper that maintains Exponential Moving Average (EMA) of model parameters.

    - Properties:
        - optimizer: The base optimizer to wrap.
        - ema_decay: The decay factor for EMA.
        - params: The original parameters tracked.
        - ema_params: The parameters that will be used for EMA.
    """
    __ema_decay: float
    base_optimizer: O
    ema_params: Iterable[Parameter]
    is_ema_parameters: bool
    params: Iterable[Parameter]

    @property
    def ema_decay(self) -> float:
        return self.__ema_decay

    @ema_decay.setter
    def ema_decay(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError("EMA decay should be in the range (0, 1).")
        self.__ema_decay = value

    def __init__(self, optimizer: O, parameters: Iterable[Parameter], *, ema_decay: float = 0.999) -> None:
        """
        Wrap a base optimizer and maintain Exponential Moving Average (EMA) of model parameters.

        - Parameters:
            - optimizer: The base optimizer to wrap in `torch.optim.optimizer.Optimizer`.
            - parameters: The model parameters to track in `list[torch.nn.Parameter]`.
            - ema_decay: The decay factor for EMA in `float`.
        """
        parameters = list(parameters)
        self.base_optimizer = optimizer
        self.ema_decay = ema_decay
        self.ema_params = copy.deepcopy(parameters)
        self.is_ema_parameters = False
        self.params = parameters

        # Create a deep copy of model parameters for EMA
        for param in self.ema_params:
            param.requires_grad = False  # EMA parameters should not be trainable
        super().__init__(self.params, optimizer.defaults)

    def __enter__(self) -> 'EMAOptimizer[O]':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Restore the original model parameters when exiting the context.
        """
        if self.is_ema_parameters:
            self.swap_parameters()

    def __getstate__(self) -> EMAState:
        optim_state = super().__getstate__()
        full_state = EMAState(optim_state=optim_state, ema_decay=self.ema_decay, ema_params=self.ema_params, base_optimizer=self.base_optimizer, params=self.params)
        return full_state

    def __setstate__(self, state: EMAState) -> None:
        super().__setstate__(state['optim_state'])
        self.ema_decay = state['ema_decay']
        self.ema_params = state['ema_params']
        self.base_optimizer = cast(O, state['base_optimizer'])
        self.params = state['params']
        self.is_ema_parameters = False

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
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
        # move ema model to the same device as the model
        for ema_param, param in zip(self.ema_params, self.params):
            ema_param.data = ema_param.data.to(param.device)

        # ema update
        with torch.no_grad():
            for ema_param, param in zip(self.ema_params, self.params):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=(1.0 - self.ema_decay))

    def state_dict(self) -> dict[str, Any]:
        """
        Save the state of both the base optimizer and the EMA parameters.

        Returns: A `dict` containing the state of the optimizer and EMA model with keys in `str`.
        """
        return {
            'ema_params': self.ema_params,
            'optimizer': self.base_optimizer.state_dict(),
            'params': self.params,
            'ema_decay': self.ema_decay,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state for both the base optimizer and the EMA parameters.

        - Parameters:
            - state_dict: The state `dict` containing both optimizer and EMA states with keys in `str`.
        """
        self.base_optimizer.load_state_dict(state_dict['optimizer'])
        self.ema_params = state_dict['ema_params']
        self.ema_decay = state_dict['ema_decay']
        self.params = state_dict['params']

    def swap_parameters(self) -> None:
        """
        Swap the model's parameters with the EMA parameters for evaluation.
        """
        for ema_param, param in zip(self.ema_params, self.params):
            param.data, ema_param.data = ema_param.data, param.data
        self.is_ema_parameters = not self.is_ema_parameters

    def use_ema_parameters(self) -> 'EMAOptimizer[O]':
        """
        Temporarily replace the model parameters with the EMA parameters.
        Use this during evaluation.
        """
        if not self.is_ema_parameters:
            self.swap_parameters()
        return self

    def zero_grad(self) -> None:
        """
        Zero the gradients of the base optimizer.
        """
        self.base_optimizer.zero_grad()
