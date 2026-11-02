"""Standalone TileLang DPA4 edge-type feature gather and sum."""

from functools import lru_cache

import torch

_BLOCK_EDGES = 4
_BLOCK_CHANNELS = 32


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang edge type features require TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def edge_type_features_forward(n_nodes, n_edges, channels, dtype=T.float32):
        @T.prim_func
        def main(
            node_features: T.Tensor((n_nodes, channels), dtype),
            src: T.Tensor((n_edges,), T.int64),
            dst: T.Tensor((n_edges,), T.int64),
            edge_keep: T.Tensor((n_edges, 1), dtype),
            edge_features: T.Tensor((n_edges, channels), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, _BLOCK_EDGES),
                T.ceildiv(channels, _BLOCK_CHANNELS),
                threads=128,
            ) as (pid_edge, pid_channel):
                for edge_offset, channel_offset in T.Parallel(
                    _BLOCK_EDGES, _BLOCK_CHANNELS
                ):
                    edge = pid_edge * _BLOCK_EDGES + edge_offset
                    channel = pid_channel * _BLOCK_CHANNELS + channel_offset
                    if edge < n_edges and channel < channels:
                        source = src[edge]
                        target = dst[edge]
                        edge_features[edge, channel] = edge_keep[edge, 0] * (
                            node_features[source, channel]
                            + node_features[target, channel]
                        )

        return main

    return edge_type_features_forward


def _check_inputs(node_features, src, dst, edge_keep):
    if node_features.ndim != 2:
        raise ValueError("node_features must have shape (n_nodes, channels)")
    if src.ndim != 1 or dst.shape != src.shape:
        raise ValueError("src and dst must have matching shape (n_edges,)")
    if edge_keep.shape != (src.shape[0], 1):
        raise ValueError("edge_keep must have shape (n_edges, 1)")
    if src.dtype != torch.int64 or dst.dtype != torch.int64:
        raise TypeError("src and dst must use torch.int64")
    if any(t.dtype != torch.float32 for t in (node_features, edge_keep)):
        raise TypeError("TileLang edge type features require float32 value tensors")
    if len({node_features.device, src.device, dst.device, edge_keep.device}) != 1:
        raise ValueError("all edge type feature inputs must share a device")
    if not node_features.is_cuda or torch.version.hip is None:
        raise RuntimeError(
            "TileLang edge type feature operator requires a HIP/DCU tensor"
        )


def _forward(node_features, src, dst, edge_keep):
    n_nodes, channels = node_features.shape
    n_edges = src.shape[0]
    edge_features = torch.empty(
        (n_edges, channels), dtype=node_features.dtype, device=node_features.device
    )
    if n_edges and channels:
        _, T = _tilelang_modules()
        _forward_factory()(n_nodes, n_edges, channels, T.float32)(
            node_features.contiguous(),
            src.contiguous(),
            dst.contiguous(),
            edge_keep.contiguous(),
            edge_features,
        )
    return edge_features


class TileLangEdgeTypeFeatures(torch.autograd.Function):
    """Build edge type features; frozen inference embeddings need no backward."""

    @staticmethod
    def forward(ctx, node_features, src, dst, edge_keep):
        _check_inputs(node_features, src, dst, edge_keep)
        return _forward(
            node_features.contiguous(),
            src.contiguous(),
            dst.contiguous(),
            edge_keep.contiguous(),
        )

    @staticmethod
    def backward(ctx, grad_edge_features):
        return None, None, None, None


def tilelang_edge_type_features(node_features, src, dst, edge_keep):
    """Gather and sum source/destination type embeddings for every edge."""
    return TileLangEdgeTypeFeatures.apply(node_features, src, dst, edge_keep)


def edge_type_features_reference(node_features, src, dst, edge_keep):
    """Differentiable PyTorch reference for edge type feature construction."""
    return (node_features[src] + node_features[dst]) * edge_keep


__all__ = [
    "TileLangEdgeTypeFeatures",
    "edge_type_features_reference",
    "tilelang_edge_type_features",
]
