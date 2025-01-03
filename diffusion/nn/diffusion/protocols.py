import torch
from typing import Protocol

from diffusion.data import DiffusionData
from diffusion.scheduling import BetaSpace
from diffusion.sde import SDE, SubVPSDE, VESDE, VPSDE

TimedData = DiffusionData
