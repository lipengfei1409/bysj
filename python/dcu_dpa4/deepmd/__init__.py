"""Opt-in integration with installed DeePMD; importing this is side-effect free."""
from .patch import DeepMDOperatorHandle, install_operators
from .runtime import configure_eager

__all__=["DeepMDOperatorHandle","install_operators","configure_eager"]
