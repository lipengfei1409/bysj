"""Megakernel (dmk) subtree: the persistent runtime and both ways into its ops.

Nothing here is wired into the top-level ``kernels`` build on purpose.  The
persistent runtime and the ordinary-launch entry points that mirror it live
under this namespace, so the main extension stays free of the VM's LDS budget,
page sizes and scheduling contract.
"""

from .so2_conv_dmk import so2_conv_dmk, so2_conv_dmk_available

__all__ = ["so2_conv_dmk", "so2_conv_dmk_available"]
