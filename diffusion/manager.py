from torchmanager_core import deprecated

from .managers import DDPMManager


@deprecated("v0.2", "v1")
class Manager(DDPMManager):
    pass
