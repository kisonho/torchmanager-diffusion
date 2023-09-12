"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc, torch
from typing import Collection


class SDE(abc.ABC):
    """
    SDE abstract class. Functions are designed for a mini-batch of inputs.

    * abstract class
    
    - Properties:
        - is_reversing: A `bool` indicating whether the SDE is reversing
        - N: A `int` indicating the number of discretization time steps
        - T: A `int` indicating the end time of the SDE

    - Methods to implement:
        - `marginal_prob`: Parameters to determine the marginal distribution of the SDE, $p_t(x)$
        - `prior_logp`: Compute log-density of the prior distribution
        - `prior_sampling`: Generate one sample from the prior distribution, $p_T(x)$
        - `__call__`: Compute drift and diffusion coefficients
    """
    is_reversing: bool
    N: int

    def __init__(self, N: int, /) -> None:
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.is_reversing = False
        self.N = N

    @property
    def T(self) -> int:
        """End time of the SDE."""
        return 1

    def discretize(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    @abc.abstractmethod
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape: Collection[int]) -> torch.Tensor:
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute drift and diffusion coefficients.

        - Parameters:
            - x: a `torch.Tensor` of the input data
            - t: a time step in `torch.Tensor`
        - Returns: A `tuple` of drift and diffusion coefficients in `torch.Tensor`
        """
        pass
