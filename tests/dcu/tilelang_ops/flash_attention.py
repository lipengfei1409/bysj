"""Test-local dense TileLang FlashAttention operator for HIP/DCU."""

import importlib.util
import math
from functools import lru_cache
from typing import Any, Optional, Tuple

import torch

_BLOCK_M = 64
_BLOCK_N = 64
_THREADS = 128


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
        from tilelang.tileop.base import GemmWarpPolicy
    except ImportError as error:
        raise RuntimeError(
            "TileLang FlashAttention requires a TileLang installation"
        ) from error
    return tilelang, T, GemmWarpPolicy


def _tilelang_dtype(dtype: torch.dtype, T: Any):
    if dtype == torch.float16:
        return T.float16
    if dtype == torch.bfloat16:
        tilelang_bfloat16 = getattr(T, "bfloat16", None)
        if tilelang_bfloat16 is None:
            raise RuntimeError("this TileLang build does not expose bfloat16")
        return tilelang_bfloat16
    raise TypeError("TileLang FlashAttention supports torch.float16 and torch.bfloat16")


def _forward_kernel_config(device: torch.device) -> tuple[int, int, int, int, bool]:
    """Use the conservative shared-layout configuration for this HIP kernel."""
    if device.type != "cuda" or torch.version.hip is None:
        raise RuntimeError("TileLang FlashAttention requires a HIP/DCU device")
    # The DCU runtime may report an architecture name that does not identify
    # RDNA, even when its fragment layouts need the same D-to-A bridge. Use the
    # AMD example's WMMA-safe tile and shared-memory path on every HIP target.
    return 32, 32, 64, 1, True


@lru_cache(maxsize=1)
def _forward_kernel_factory():
    tilelang, T, GemmWarpPolicy = _tilelang_modules()

    @tilelang.jit(target="hip", out_idx=[3, 4])
    def flash_attention_forward(
        batch,
        q_len,
        kv_len,
        heads,
        head_dim,
        causal,
        scale,
        block_m,
        block_n,
        threads,
        k_pack,
        use_shared_prob_bridge,
        dtype=T.float16,
    ):
        q_shape = [batch, q_len, heads, head_dim]
        kv_shape = [batch, kv_len, heads, head_dim]

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
            LSE: T.Tensor([batch, heads, q_len], T.float32),
        ):
            with T.Kernel(
                T.ceildiv(q_len, block_m), batch * heads, threads=threads
            ) as (pid_q, pid_bh):
                batch_id = pid_bh // heads
                head_id = pid_bh % heads
                q_offset = pid_q * block_m

                q_shared = T.alloc_shared([block_m, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                v_shared = T.alloc_shared([block_n, head_dim], dtype)
                if use_shared_prob_bridge:
                    probs_shared = T.alloc_shared([block_m, block_n], dtype)
                scores = T.alloc_fragment([block_m, block_n], T.float32)
                probs = T.alloc_fragment([block_m, block_n], T.float32)
                probs_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, head_dim], T.float32)
                m_i = T.alloc_fragment([block_m], T.float32)
                m_prev = T.alloc_fragment([block_m], T.float32)
                m_block = T.alloc_fragment([block_m], T.float32)
                l_i = T.alloc_fragment([block_m], T.float32)
                row_sum = T.alloc_fragment([block_m], T.float32)
                scale_factor = T.alloc_fragment([block_m], T.float32)
                T.fill(acc_o, 0.0)
                T.fill(m_i, -T.infinity(T.float32))
                T.fill(l_i, 0.0)

                for i, j in T.Parallel(block_m, head_dim):
                    q_index = q_offset + i
                    if q_index < q_len:
                        q_shared[i, j] = Q[batch_id, q_index, head_id, j]
                    else:
                        q_shared[i, j] = 0.0

                for pid_k in T.serial(T.ceildiv(kv_len, block_n)):
                    k_offset = pid_k * block_n
                    should_compute = (not causal) or (
                        k_offset <= q_offset + block_m - 1 + (kv_len - q_len)
                    )
                    if should_compute:
                        for i, j in T.Parallel(block_n, head_dim):
                            k_index = k_offset + i
                            if k_index < kv_len:
                                k_shared[i, j] = K[batch_id, k_index, head_id, j]
                                v_shared[i, j] = V[batch_id, k_index, head_id, j]
                            else:
                                k_shared[i, j] = 0.0
                                v_shared[i, j] = 0.0

                        T.clear(scores)
                        T.gemm(
                            q_shared,
                            k_shared,
                            scores,
                            transpose_B=True,
                            k_pack=k_pack,
                            policy=GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            q_index = q_offset + i
                            k_index = k_offset + j
                            if q_index < q_len and k_index < kv_len:
                                if causal:
                                    if k_index <= q_index + (kv_len - q_len):
                                        scores[i, j] = scores[i, j] * scale
                                    else:
                                        scores[i, j] = -T.infinity(T.float32)
                                else:
                                    scores[i, j] = scores[i, j] * scale
                            else:
                                scores[i, j] = -T.infinity(T.float32)

                        T.fill(m_block, -T.infinity(T.float32))
                        T.reduce_max(scores, m_block, dim=1, clear=False)
                        T.copy(m_i, m_prev)
                        for i in T.Parallel(block_m):
                            m_i[i] = T.max(m_prev[i], m_block[i])
                            if m_prev[i] > -1.0e30:
                                scale_factor[i] = T.exp(m_prev[i] - m_i[i])
                            else:
                                scale_factor[i] = 0.0
                            l_i[i] *= scale_factor[i]
                        for i, j in T.Parallel(block_m, head_dim):
                            acc_o[i, j] *= scale_factor[i]
                        for i, j in T.Parallel(block_m, block_n):
                            if scores[i, j] > -1.0e30:
                                probs[i, j] = T.exp(scores[i, j] - m_i[i])
                            else:
                                probs[i, j] = 0.0
                        T.reduce_sum(probs, row_sum, dim=1)
                        for i in T.Parallel(block_m):
                            l_i[i] += row_sum[i]
                        if use_shared_prob_bridge:
                            for i, j in T.Parallel(block_m, block_n):
                                probs_shared[i, j] = T.cast(probs[i, j], dtype)
                            T.copy(probs_shared, probs_cast)
                        else:
                            T.copy(probs, probs_cast)
                        T.gemm(
                            probs_cast,
                            v_shared,
                            acc_o,
                            k_pack=k_pack,
                            policy=GemmWarpPolicy.FullRow,
                        )

                for i, j in T.Parallel(block_m, head_dim):
                    q_index = q_offset + i
                    if q_index < q_len:
                        if l_i[i] > 0.0:
                            Output[batch_id, q_index, head_id, j] = acc_o[i, j] / l_i[i]
                        else:
                            Output[batch_id, q_index, head_id, j] = 0.0
                for i in T.Parallel(block_m):
                    q_index = q_offset + i
                    if q_index < q_len:
                        if l_i[i] > 0.0:
                            LSE[batch_id, head_id, q_index] = T.log(l_i[i]) + m_i[i]
                        else:
                            LSE[batch_id, head_id, q_index] = -T.infinity(T.float32)

        return main

    return flash_attention_forward


@lru_cache(maxsize=1)
def _delta_kernel_factory():
    tilelang, T, _ = _tilelang_modules()

    @tilelang.jit(target="hip", out_idx=[2])
    def flash_attention_delta(batch, q_len, heads, head_dim, block_m, dtype=T.float16):
        q_shape = [batch, q_len, heads, head_dim]

        @T.prim_func
        def main(
            Output: T.Tensor(q_shape, dtype),
            dOutput: T.Tensor(q_shape, dtype),
            Delta: T.Tensor([batch, heads, q_len], T.float32),
        ):
            with T.Kernel(
                T.ceildiv(q_len, block_m), batch * heads, threads=128
            ) as (pid_q, pid_bh):
                batch_id = pid_bh // heads
                head_id = pid_bh % heads
                q_offset = pid_q * block_m
                products = T.alloc_fragment([block_m, head_dim], T.float32)
                delta = T.alloc_fragment([block_m], T.float32)
                for i, j in T.Parallel(block_m, head_dim):
                    q_index = q_offset + i
                    if q_index < q_len:
                        products[i, j] = T.Cast(
                            T.float32, Output[batch_id, q_index, head_id, j]
                        ) * T.Cast(T.float32, dOutput[batch_id, q_index, head_id, j])
                    else:
                        products[i, j] = 0.0
                T.reduce_sum(products, delta, dim=1)
                for i in T.Parallel(block_m):
                    q_index = q_offset + i
                    if q_index < q_len:
                        Delta[batch_id, head_id, q_index] = delta[i]

        return main

    return flash_attention_delta


@lru_cache(maxsize=1)
def _backward_q_kernel_factory():
    tilelang, T, GemmWarpPolicy = _tilelang_modules()

    @tilelang.jit(target="hip")
    def flash_attention_backward_q(
        batch,
        q_len,
        kv_len,
        heads,
        head_dim,
        causal,
        scale,
        block_m,
        block_n,
        threads,
        dtype=T.float16,
    ):
        q_shape = [batch, q_len, heads, head_dim]
        kv_shape = [batch, kv_len, heads, head_dim]
        stats_shape = [batch, heads, q_len]

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            dOutput: T.Tensor(q_shape, dtype),
            LSE: T.Tensor(stats_shape, T.float32),
            Delta: T.Tensor(stats_shape, T.float32),
            dQ: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(q_len, block_m), batch * heads, threads=threads
            ) as (pid_q, pid_bh):
                batch_id = pid_bh // heads
                head_id = pid_bh % heads
                q_offset = pid_q * block_m
                q_shared = T.alloc_shared([block_m, head_dim], dtype)
                do_shared = T.alloc_shared([block_m, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                v_shared = T.alloc_shared([block_n, head_dim], dtype)
                scores = T.alloc_fragment([block_m, block_n], T.float32)
                dprobs = T.alloc_fragment([block_m, block_n], T.float32)
                probs = T.alloc_fragment([block_m, block_n], T.float32)
                dscore = T.alloc_fragment([block_m, block_n], T.float32)
                probs_cast = T.alloc_fragment([block_m, block_n], dtype)
                dscore_cast = T.alloc_fragment([block_m, block_n], dtype)
                dq_acc = T.alloc_fragment([block_m, head_dim], T.float32)
                T.fill(dq_acc, 0.0)
                for i, j in T.Parallel(block_m, head_dim):
                    q_index = q_offset + i
                    if q_index < q_len:
                        q_shared[i, j] = Q[batch_id, q_index, head_id, j]
                        do_shared[i, j] = dOutput[batch_id, q_index, head_id, j]
                    else:
                        q_shared[i, j] = 0.0
                        do_shared[i, j] = 0.0

                for pid_k in T.serial(T.ceildiv(kv_len, block_n)):
                    k_offset = pid_k * block_n
                    for i, j in T.Parallel(block_n, head_dim):
                        k_index = k_offset + i
                        if k_index < kv_len:
                            k_shared[i, j] = K[batch_id, k_index, head_id, j]
                            v_shared[i, j] = V[batch_id, k_index, head_id, j]
                        else:
                            k_shared[i, j] = 0.0
                            v_shared[i, j] = 0.0
                    T.clear(scores)
                    T.gemm(
                        q_shared,
                        k_shared,
                        scores,
                        transpose_B=True,
                        k_pack=2,
                        policy=GemmWarpPolicy.FullRow,
                    )
                    T.clear(dprobs)
                    T.gemm(
                        do_shared,
                        v_shared,
                        dprobs,
                        transpose_B=True,
                        k_pack=2,
                        policy=GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(block_m, block_n):
                        q_index = q_offset + i
                        k_index = k_offset + j
                        valid = q_index < q_len and k_index < kv_len
                        if causal:
                            valid = valid and (
                                k_index <= q_index + (kv_len - q_len)
                            )
                        if valid:
                            lse = LSE[batch_id, head_id, q_index]
                            if lse > -1.0e30:
                                probs[i, j] = T.exp(scores[i, j] * scale - lse)
                            else:
                                probs[i, j] = 0.0
                            dscore[i, j] = probs[i, j] * (
                                dprobs[i, j] - Delta[batch_id, head_id, q_index]
                            ) * scale
                        else:
                            probs[i, j] = 0.0
                            dscore[i, j] = 0.0
                    T.copy(dscore, dscore_cast)
                    T.gemm(
                        dscore_cast,
                        k_shared,
                        dq_acc,
                        k_pack=2,
                        policy=GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_m, head_dim):
                    q_index = q_offset + i
                    if q_index < q_len:
                        dQ[batch_id, q_index, head_id, j] = dq_acc[i, j]

        return main

    return flash_attention_backward_q


@lru_cache(maxsize=1)
def _backward_kv_kernel_factory():
    tilelang, T, GemmWarpPolicy = _tilelang_modules()

    @tilelang.jit(target="hip")
    def flash_attention_backward_kv(
        batch,
        q_len,
        kv_len,
        heads,
        head_dim,
        causal,
        scale,
        block_m,
        block_n,
        threads,
        dtype=T.float16,
    ):
        q_shape = [batch, q_len, heads, head_dim]
        kv_shape = [batch, kv_len, heads, head_dim]
        stats_shape = [batch, heads, q_len]

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            dOutput: T.Tensor(q_shape, dtype),
            LSE: T.Tensor(stats_shape, T.float32),
            Delta: T.Tensor(stats_shape, T.float32),
            dK: T.Tensor(kv_shape, dtype),
            dV: T.Tensor(kv_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(kv_len, block_n), batch * heads, threads=threads
            ) as (pid_k, pid_bh):
                batch_id = pid_bh // heads
                head_id = pid_bh % heads
                k_offset = pid_k * block_n
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                v_shared = T.alloc_shared([block_n, head_dim], dtype)
                q_shared = T.alloc_shared([block_m, head_dim], dtype)
                do_shared = T.alloc_shared([block_m, head_dim], dtype)
                scores = T.alloc_fragment([block_m, block_n], T.float32)
                dprobs = T.alloc_fragment([block_m, block_n], T.float32)
                probs = T.alloc_fragment([block_m, block_n], T.float32)
                dscore = T.alloc_fragment([block_m, block_n], T.float32)
                probs_cast = T.alloc_fragment([block_m, block_n], dtype)
                dscore_cast = T.alloc_fragment([block_m, block_n], dtype)
                dk_acc = T.alloc_fragment([block_n, head_dim], T.float32)
                dv_acc = T.alloc_fragment([block_n, head_dim], T.float32)
                T.fill(dk_acc, 0.0)
                T.fill(dv_acc, 0.0)
                for i, j in T.Parallel(block_n, head_dim):
                    k_index = k_offset + i
                    if k_index < kv_len:
                        k_shared[i, j] = K[batch_id, k_index, head_id, j]
                        v_shared[i, j] = V[batch_id, k_index, head_id, j]
                    else:
                        k_shared[i, j] = 0.0
                        v_shared[i, j] = 0.0

                for pid_q in T.serial(T.ceildiv(q_len, block_m)):
                    q_offset = pid_q * block_m
                    for i, j in T.Parallel(block_m, head_dim):
                        q_index = q_offset + i
                        if q_index < q_len:
                            q_shared[i, j] = Q[batch_id, q_index, head_id, j]
                            do_shared[i, j] = dOutput[batch_id, q_index, head_id, j]
                        else:
                            q_shared[i, j] = 0.0
                            do_shared[i, j] = 0.0
                    T.clear(scores)
                    T.gemm(
                        q_shared,
                        k_shared,
                        scores,
                        transpose_B=True,
                        k_pack=2,
                        policy=GemmWarpPolicy.FullRow,
                    )
                    T.clear(dprobs)
                    T.gemm(
                        do_shared,
                        v_shared,
                        dprobs,
                        transpose_B=True,
                        k_pack=2,
                        policy=GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(block_m, block_n):
                        q_index = q_offset + i
                        k_index = k_offset + j
                        valid = q_index < q_len and k_index < kv_len
                        if causal:
                            valid = valid and (
                                k_index <= q_index + (kv_len - q_len)
                            )
                        if valid:
                            lse = LSE[batch_id, head_id, q_index]
                            if lse > -1.0e30:
                                probs[i, j] = T.exp(scores[i, j] * scale - lse)
                            else:
                                probs[i, j] = 0.0
                            dscore[i, j] = probs[i, j] * (
                                dprobs[i, j] - Delta[batch_id, head_id, q_index]
                            ) * scale
                        else:
                            probs[i, j] = 0.0
                            dscore[i, j] = 0.0
                    T.copy(probs, probs_cast)
                    T.copy(dscore, dscore_cast)
                    T.gemm(
                        probs_cast,
                        do_shared,
                        dv_acc,
                        transpose_A=True,
                        k_pack=2,
                        policy=GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        dscore_cast,
                        q_shared,
                        dk_acc,
                        transpose_A=True,
                        k_pack=2,
                        policy=GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_n, head_dim):
                    k_index = k_offset + i
                    if k_index < kv_len:
                        dK[batch_id, k_index, head_id, j] = dk_acc[i, j]
                        dV[batch_id, k_index, head_id, j] = dv_acc[i, j]

        return main

    return flash_attention_backward_kv


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: Optional[float],
    causal: bool,
    window_size: Tuple[int, int],
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_attn_probs: bool,
) -> float:
    if dropout_p != 0.0:
        raise NotImplementedError(
            "TileLang FlashAttention currently requires dropout_p=0"
        )
    if window_size not in ((-1, -1), None):
        raise NotImplementedError(
            "TileLang FlashAttention does not support windowed attention"
        )
    if softcap != 0.0:
        raise NotImplementedError("TileLang FlashAttention does not support softcap")
    if alibi_slopes is not None:
        raise NotImplementedError(
            "TileLang FlashAttention does not support alibi_slopes"
        )
    if return_attn_probs:
        raise NotImplementedError(
            "TileLang FlashAttention does not return attention probabilities"
        )
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "q, k, and v must have shape (batch, sequence, heads, head_dim)"
        )
    if q.shape[0] != k.shape[0] or k.shape != v.shape:
        raise ValueError("q, k, and v must have matching batch and key/value shapes")
    if q.shape[2:] != k.shape[2:]:
        raise ValueError(
            "TileLang FlashAttention requires matching Q/K/V head counts and dimensions"
        )
    if (
        q.shape[0] < 1
        or q.shape[1] < 1
        or k.shape[1] < 1
        or q.shape[2] < 1
        or q.shape[3] < 1
    ):
        raise ValueError("batch, sequence, heads, and head_dim must be non-empty")
    if q.shape[3] > 128 or q.shape[3] % 16 != 0:
        raise ValueError("head_dim must be a multiple of 16 no larger than 128")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("TileLang FlashAttention requires float16 or bfloat16 inputs")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, and v must be contiguous")
    if not q.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang FlashAttention requires a HIP/DCU tensor")
    if importlib.util.find_spec("tilelang") is None:
        raise RuntimeError("TileLang FlashAttention requires TileLang to be installed")
    scale = (
        1.0 / math.sqrt(q.shape[-1])
        if softmax_scale is None
        else float(softmax_scale)
    )
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")
    return scale


class _TileLangFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal):
        _, T, _ = _tilelang_modules()
        dtype = _tilelang_dtype(q.dtype, T)
        batch, q_len, heads, head_dim = q.shape
        kv_len = k.shape[1]
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        block_m, block_n, threads, k_pack, use_shared_prob_bridge = (
            _forward_kernel_config(q.device)
        )
        output, lse = _forward_kernel_factory()(
            batch,
            q_len,
            kv_len,
            heads,
            head_dim,
            bool(causal),
            float(scale),
            block_m,
            block_n,
            threads,
            k_pack,
            use_shared_prob_bridge,
            dtype,
        )(q, k, v)
        ctx.scale = float(scale)
        ctx.causal = bool(causal)
        ctx.save_for_backward(q, k, v, output, lse)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        batch, q_len, heads, head_dim = q.shape
        kv_len = k.shape[1]
        _, T, _ = _tilelang_modules()
        dtype = _tilelang_dtype(q.dtype, T)
        grad_output = grad_output.contiguous()
        delta = _delta_kernel_factory()(
            batch, q_len, heads, head_dim, _BLOCK_M, dtype
        )(output, grad_output)
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        _backward_q_kernel_factory()(
            batch,
            q_len,
            kv_len,
            heads,
            head_dim,
            ctx.causal,
            ctx.scale,
            _BLOCK_M,
            _BLOCK_N,
            _THREADS,
            dtype,
        )(q, k, v, grad_output, lse, delta, grad_q)
        _backward_kv_kernel_factory()(
            batch,
            q_len,
            kv_len,
            heads,
            head_dim,
            ctx.causal,
            ctx.scale,
            _BLOCK_M,
            _BLOCK_N,
            _THREADS,
            dtype,
        )(q, k, v, grad_output, lse, delta, grad_k, grad_v)
        return grad_q, grad_k, grad_v, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[Tuple[int, int]] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """Dense FlashAttention with a test-local HIP TileLang implementation.

    The signature follows the common open-source ``flash_attn_func`` entry
    point. This first DCU kernel supports MHA, zero dropout, optional scaling,
    and causal masking; unsupported optional modes raise rather than silently
    falling back. Inputs are contiguous BSHD FP16/BF16 tensors with matching
    heads and a head dimension divisible by 16 (at most 128). Query and KV
    sequence lengths may differ; causal masking uses bottom-right alignment.
    ``deterministic`` is accepted for signature compatibility; this
    implementation has deterministic ownership for each gradient tile.
    """
    del deterministic
    scale = _validate_inputs(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        return_attn_probs,
    )
    return _TileLangFlashAttention.apply(q, k, v, scale, causal)


__all__ = ["flash_attn_func"]
