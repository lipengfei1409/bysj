"""PyTorch reference implementations for explicit testing and comparison."""

from .gie import gie_zonal_scatter_reference, zonal_scatter_reference
from .rmsnorm import rmsnorm
from .so2_conv_dmk import (
    so2_conv_dmk_kernel_order,
    so2_conv_dmk_reference,
)
from .slot import slot_pack_scale_reference, slot_reduce_reference
from .segment import (
    force_assembly_reference,
    segment_softmax_reference,
    segment_sum_reference,
)

__all__ = [
    "force_assembly_reference",
    "gie_zonal_scatter_reference",
    "rmsnorm",
    "segment_softmax_reference",
    "segment_sum_reference",
    "slot_pack_scale_reference",
    "so2_conv_dmk_kernel_order",
    "so2_conv_dmk_reference",
    "slot_reduce_reference",
    "zonal_scatter_reference",
]
