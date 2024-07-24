from enum import Enum

from .base import SDE
from .ve import VESDE
from .vp import SubVPSDE, VPSDE


class SDEType(Enum):
    """The supported SDE types."""
    VE = VESDE
    VP = VPSDE
    SUB_VP = SubVPSDE

    def load(self, N: int) -> SDE:
        """
        Load the SDE.

        - Parameters:
            - N: An `int` of the number of discretization time steps.
        - Returns: An `SDE` instance.
        """
        return self.value(N)
