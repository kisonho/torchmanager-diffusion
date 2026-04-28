from packaging.version import Version

API = Version("v1.3")
CURRENT = Version("v1.3a4")
DESCRIPTION = f"Torchmanager Implementation for Diffusion Model ({CURRENT})"

__all__ = ["API", "CURRENT", "DESCRIPTION"]
