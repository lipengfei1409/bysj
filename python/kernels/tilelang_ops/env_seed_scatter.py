"""Standalone TileLang environment-seed outer-product scatter."""

from functools import lru_cache

import torch

_BLOCK_EDGES = 4
_BLOCK_FEATURES = 32


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang environment-seed scatter requires TileLang to be installed"
        ) from error
    return tilelang, T


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def env_seed_forward(
        n_edges,
        n_nodes,
        coord_dim,
        embed_dim,
        block_edges,
        block_features,
        dtype=T.float32,
    ):
        n_features = coord_dim * embed_dim

        @T.prim_func
        def main(
            r_tilde: T.Tensor((n_edges, coord_dim), dtype),
            features: T.Tensor((n_edges, embed_dim), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            inv_sqrt_deg: T.Tensor((n_nodes,), dtype),
            output: T.Tensor((n_nodes, coord_dim, embed_dim), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(n_features, block_features),
                threads=128,
            ) as (pid_edge, pid_feature):
                for edge_offset, feature_offset in T.Parallel(
                    block_edges, block_features
                ):
                    edge = pid_edge * block_edges + edge_offset
                    feature = pid_feature * block_features + feature_offset
                    if edge < n_edges and feature < n_features:
                        coord = feature // embed_dim
                        embed = feature - coord * embed_dim
                        node = dst[edge]
                        value = (
                            r_tilde[edge, coord]
                            * features[edge, embed]
                            * inv_sqrt_deg[node]
                        )
                        T.atomic_add(output[node, coord, embed], value)

        return main

    return env_seed_forward


@lru_cache(maxsize=1)
def _backward_geometry_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def env_seed_backward_geometry(
        n_edges, n_nodes, coord_dim, embed_dim, block_edges, dtype=T.float32
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_nodes, coord_dim, embed_dim), dtype),
            features: T.Tensor((n_edges, embed_dim), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            inv_sqrt_deg: T.Tensor((n_nodes,), dtype),
            grad_r_tilde: T.Tensor((n_edges, coord_dim), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), coord_dim, threads=128) as (
                pid_edge,
                coord,
            ):
                for edge_offset in T.Parallel(block_edges):
                    edge = pid_edge * block_edges + edge_offset
                    if edge < n_edges:
                        node = dst[edge]
                        total = 0.0
                        for embed in T.serial(embed_dim):
                            total += (
                                grad_output[node, coord, embed] * features[edge, embed]
                            )
                        grad_r_tilde[edge, coord] = total * inv_sqrt_deg[node]

        return main

    return env_seed_backward_geometry


@lru_cache(maxsize=1)
def _backward_features_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def env_seed_backward_features(
        n_edges,
        n_nodes,
        coord_dim,
        embed_dim,
        block_edges,
        block_features,
        dtype=T.float32,
    ):
        @T.prim_func
        def main(
            grad_output: T.Tensor((n_nodes, coord_dim, embed_dim), dtype),
            r_tilde: T.Tensor((n_edges, coord_dim), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            inv_sqrt_deg: T.Tensor((n_nodes,), dtype),
            grad_features: T.Tensor((n_edges, embed_dim), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(embed_dim, block_features),
                threads=128,
            ) as (pid_edge, pid_embed):
                for edge_offset, embed_offset in T.Parallel(
                    block_edges, block_features
                ):
                    edge = pid_edge * block_edges + edge_offset
                    embed = pid_embed * block_features + embed_offset
                    if edge < n_edges and embed < embed_dim:
                        node = dst[edge]
                        total = 0.0
                        for coord in T.serial(coord_dim):
                            total += (
                                grad_output[node, coord, embed] * r_tilde[edge, coord]
                            )
                        grad_features[edge, embed] = total * inv_sqrt_deg[node]

        return main

    return env_seed_backward_features


@lru_cache(maxsize=1)
def _backward_degree_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def env_seed_backward_degree(
        n_edges,
        n_nodes,
        coord_dim,
        embed_dim,
        block_edges,
        block_features,
        dtype=T.float32,
    ):
        n_features = coord_dim * embed_dim

        @T.prim_func
        def main(
            grad_output: T.Tensor((n_nodes, coord_dim, embed_dim), dtype),
            r_tilde: T.Tensor((n_edges, coord_dim), dtype),
            features: T.Tensor((n_edges, embed_dim), dtype),
            dst: T.Tensor((n_edges,), T.int64),
            grad_inv_sqrt_deg: T.Tensor((n_nodes,), dtype),
        ):
            with T.Kernel(
                T.ceildiv(n_edges, block_edges),
                T.ceildiv(n_features, block_features),
                threads=128,
            ) as (pid_edge, pid_feature):
                for edge_offset, feature_offset in T.Parallel(
                    block_edges, block_features
                ):
                    edge = pid_edge * block_edges + edge_offset
                    feature = pid_feature * block_features + feature_offset
                    if edge < n_edges and feature < n_features:
                        coord = feature // embed_dim
                        embed = feature - coord * embed_dim
                        node = dst[edge]
                        value = (
                            grad_output[node, coord, embed]
                            * r_tilde[edge, coord]
                            * features[edge, embed]
                        )
                        T.atomic_add(grad_inv_sqrt_deg[node], value)

        return main

    return env_seed_backward_degree


def _check_inputs(r_tilde, features, dst, inv_sqrt_deg):
    if r_tilde.ndim != 2:
        raise ValueError("r_tilde must have shape (n_edges, coord_dim)")
    if features.ndim != 2 or features.shape[0] != r_tilde.shape[0]:
        raise ValueError("features must have shape (n_edges, embed_dim)")
    if dst.shape != (r_tilde.shape[0],) or dst.dtype != torch.int64:
        raise TypeError("dst must be an int64 tensor with shape (n_edges,)")
    if inv_sqrt_deg.ndim not in (1, 3) or (
        inv_sqrt_deg.ndim == 3 and inv_sqrt_deg.shape[1:] != (1, 1)
    ):
        raise ValueError("inv_sqrt_deg must have shape (n_nodes,) or (n_nodes, 1, 1)")
    if any(t.dtype != torch.float32 for t in (r_tilde, features, inv_sqrt_deg)):
        raise TypeError("TileLang environment-seed scatter requires float32 values")
    if len({r_tilde.device, features.device, dst.device, inv_sqrt_deg.device}) != 1:
        raise ValueError("all environment-seed scatter inputs must share a device")
    if not r_tilde.is_cuda or torch.version.hip is None:
        raise RuntimeError(
            "TileLang environment-seed scatter requires a HIP/DCU tensor"
        )


def _flat_degree(inv_sqrt_deg):
    return inv_sqrt_deg.reshape(-1).contiguous()


def _forward(r_tilde, features, dst, inv_sqrt_deg):
    _check_inputs(r_tilde, features, dst, inv_sqrt_deg)
    n_edges, coord_dim = r_tilde.shape
    embed_dim = features.shape[1]
    n_nodes = inv_sqrt_deg.shape[0]
    output = torch.zeros(
        (n_nodes, coord_dim, embed_dim), dtype=r_tilde.dtype, device=r_tilde.device
    )
    if n_edges:
        _, T = _tilelang_modules()
        _forward_factory()(
            n_edges,
            n_nodes,
            coord_dim,
            embed_dim,
            _BLOCK_EDGES,
            _BLOCK_FEATURES,
            T.float32,
        )(
            r_tilde.contiguous(),
            features.contiguous(),
            dst.contiguous(),
            _flat_degree(inv_sqrt_deg),
            output,
        )
    return output


def _backward(grad_output, r_tilde, features, dst, inv_sqrt_deg):
    n_edges, coord_dim = r_tilde.shape
    embed_dim = features.shape[1]
    n_nodes = inv_sqrt_deg.shape[0]
    grad_r_tilde = torch.empty_like(r_tilde)
    grad_features = torch.empty_like(features)
    grad_inv_sqrt_deg = torch.zeros_like(_flat_degree(inv_sqrt_deg))
    if n_edges:
        _, T = _tilelang_modules()
        common = (n_edges, n_nodes, coord_dim, embed_dim)
        _backward_geometry_factory()(*common, _BLOCK_EDGES, T.float32)(
            grad_output.contiguous(),
            features.contiguous(),
            dst.contiguous(),
            _flat_degree(inv_sqrt_deg),
            grad_r_tilde,
        )
        _backward_features_factory()(*common, _BLOCK_EDGES, _BLOCK_FEATURES, T.float32)(
            grad_output.contiguous(),
            r_tilde.contiguous(),
            dst.contiguous(),
            _flat_degree(inv_sqrt_deg),
            grad_features,
        )
        _backward_degree_factory()(*common, _BLOCK_EDGES, _BLOCK_FEATURES, T.float32)(
            grad_output.contiguous(),
            r_tilde.contiguous(),
            features.contiguous(),
            dst.contiguous(),
            grad_inv_sqrt_deg,
        )
    return grad_r_tilde, grad_features, grad_inv_sqrt_deg.reshape(inv_sqrt_deg.shape)


class TileLangEnvironmentSeedScatter(torch.autograd.Function):
    """Scatter ``r_tilde outer features`` and normalize per destination node."""

    @staticmethod
    def forward(ctx, r_tilde, features, dst, inv_sqrt_deg):
        _check_inputs(r_tilde, features, dst, inv_sqrt_deg)
        r_tilde, features, dst, inv_sqrt_deg = (
            r_tilde.contiguous(),
            features.contiguous(),
            dst.contiguous(),
            inv_sqrt_deg.contiguous(),
        )
        ctx.save_for_backward(r_tilde, features, dst, inv_sqrt_deg)
        return _forward(r_tilde, features, dst, inv_sqrt_deg)

    @staticmethod
    def backward(ctx, grad_output):
        r_tilde, features, dst, inv_sqrt_deg = ctx.saved_tensors
        if not any(ctx.needs_input_grad[:2]) and not ctx.needs_input_grad[3]:
            return None, None, None, None
        grad_r_tilde, grad_features, grad_inv_sqrt_deg = _backward(
            grad_output, r_tilde, features, dst, inv_sqrt_deg
        )
        return (
            grad_r_tilde if ctx.needs_input_grad[0] else None,
            grad_features if ctx.needs_input_grad[1] else None,
            None,
            grad_inv_sqrt_deg if ctx.needs_input_grad[3] else None,
        )


def tilelang_environment_seed_scatter(r_tilde, features, dst, inv_sqrt_deg):
    """Aggregate the DPA4 environment-seed outer product by destination."""
    return TileLangEnvironmentSeedScatter.apply(r_tilde, features, dst, inv_sqrt_deg)


def environment_seed_scatter_reference(r_tilde, features, dst, inv_sqrt_deg):
    """Differentiable PyTorch reference for the environment-seed scatter."""
    if inv_sqrt_deg.ndim == 3:
        inv_sqrt_deg = inv_sqrt_deg.reshape(-1)
    messages = r_tilde[:, :, None] * features[:, None, :]
    messages = messages * inv_sqrt_deg.index_select(0, dst)[:, None, None]
    output = r_tilde.new_zeros(
        (inv_sqrt_deg.shape[0], r_tilde.shape[1], features.shape[1])
    )
    if r_tilde.shape[0]:
        output.index_add_(0, dst, messages)
    return output


__all__ = [
    "TileLangEnvironmentSeedScatter",
    "environment_seed_scatter_reference",
    "tilelang_environment_seed_scatter",
]
