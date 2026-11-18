"""PyTorch reference implementations for explicit testing and comparison."""

from .gie import gie_zonal_scatter_reference, zonal_scatter_reference
from .rmsnorm import rmsnorm
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
    "zonal_scatter_reference",
]
