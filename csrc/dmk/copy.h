#pragma once

#include "tensor.h"
#include "dumma.h"

namespace dmk::copy {

using tensor::MemorySpace;
using tensor::Shape;
using tensor::StridedLayout;
using tensor::Tensor;

// Adjacent worker IDs should access adjacent global addresses. An axis with
// the smallest element stride varies first; padded rows may still leave gaps.
namespace detail {
template <size_t Rank>
DMK_HOST_DEVICE inline void row_major_order(size_t (&axes)[Rank]) {
    for (size_t index = 0; index < Rank; ++index)
        axes[index] = Rank - 1 - index;
}

template <size_t Rank>
DMK_HOST_DEVICE inline void stride_order(const StridedLayout<Rank>& layout,
                                         size_t (&axes)[Rank]) {
    for (size_t index = 0; index < Rank; ++index) axes[index] = index;
    for (size_t index = 1; index < Rank; ++index) {
        const size_t axis = axes[index];
        size_t position = index;
        while (position > 0 &&
               layout.strides[axes[position - 1]] > layout.strides[axis]) {
            axes[position] = axes[position - 1];
            --position;
        }
        axes[position] = axis;
    }
}

template <typename DstT, size_t Rank, MemorySpace DstSpace, typename DstLayout,
          typename SrcT, MemorySpace SrcSpace, typename SrcLayout>
DMK_HOST_DEVICE inline bool copy_ordered(
    Tensor<DstT, Rank, DstSpace, DstLayout> dst,
    Tensor<SrcT, Rank, SrcSpace, SrcLayout> src,
    size_t worker_id, size_t worker_count, const size_t (&axes)[Rank]) {
    static_assert(DstSpace != MemorySpace::Register &&
                  SrcSpace != MemorySpace::Register,
                  "Register fragments require an instruction-specific copy");
    static_assert(!is_const<DstT>::value, "Destination must be writable");
    static_assert(is_same<DstT, remove_const_t<SrcT>>::value,
                  "Source and destination element types must match");

    if (worker_count == 0 || worker_id >= worker_count) return false;
    for (size_t axis = 0; axis < Rank; ++axis)
        if (dst.shape.dims[axis] != src.shape.dims[axis]) return false;

    size_t count = 0;
    if (!src.shape.try_numel(count)) return false;
    if (count == 0) return true;
    if (!src.data || !dst.data) return false;

    for (size_t linear = worker_id; linear < count;) {
        Shape<Rank> indices{};
        size_t remaining = linear;
        for (size_t position = 0; position < Rank; ++position) {
            const size_t axis = axes[position];
            const size_t extent = src.shape.dims[axis];
            indices.dims[axis] = remaining % extent;
            remaining /= extent;
        }
        dst.data[dst.offset(indices)] = src.data[src.offset(indices)];
        if (worker_count > count - 1 - linear) break;
        linear += worker_count;
    }
    return true;
}

template <typename T>
__device__ inline bool valid_row_major_lds(
    const Shape<2>& fragment_shape,
    Tensor<T, 2, MemorySpace::Shmem, StridedLayout<2>> tile) {
    if (!tile.data || !tile.shape.dims[0] || !tile.shape.dims[1]) return false;
    if (tile.shape.dims[0] != fragment_shape.dims[0] ||
        tile.shape.dims[1] != fragment_shape.dims[1]) return false;
    return tile.layout.strides[1] == 1 &&
           tile.layout.strides[0] >= tile.shape.dims[1] &&
           tile.layout.strides[0] <= static_cast<size_t>(~0u);
}

}  // namespace detail

// Generic same-shape copy traverses logical row-major order. The tensors must
// not overlap, and the caller synchronizes before another group reads dst.
template <typename DstT, size_t Rank, MemorySpace DstSpace, typename DstLayout,
          typename SrcT, MemorySpace SrcSpace, typename SrcLayout>
DMK_HOST_DEVICE inline bool tensor_copy(
    Tensor<DstT, Rank, DstSpace, DstLayout> dst,
    Tensor<SrcT, Rank, SrcSpace, SrcLayout> src,
    size_t worker_id, size_t worker_count) {
    size_t axes[Rank];
    detail::row_major_order(axes);
    return detail::copy_ordered(dst, src, worker_id, worker_count, axes);
}

// Use the global tensor's physical strides to assign consecutive addresses to
// adjacent workers. For a wave, pass lane ID and wave size; for a full block,
// pass thread ID and block size. The caller provides any required barrier.
template <typename DstT, size_t Rank, typename DstLayout,
          typename SrcT, typename SrcLayout>
DMK_HOST_DEVICE inline bool global_to_lds(
    Tensor<DstT, Rank, MemorySpace::Shmem, DstLayout> dst,
    Tensor<SrcT, Rank, MemorySpace::HBM, SrcLayout> src,
    size_t worker_id, size_t worker_count) {
    size_t axes[Rank];
    if constexpr (is_same<SrcLayout, StridedLayout<Rank>>::value)
        detail::stride_order(src.layout, axes);
    else
        detail::row_major_order(axes);
    return detail::copy_ordered(dst, src, worker_id, worker_count, axes);
}

template <typename DstT, size_t Rank, typename DstLayout,
          typename SrcT, typename SrcLayout>
DMK_HOST_DEVICE inline bool lds_to_global(
    Tensor<DstT, Rank, MemorySpace::HBM, DstLayout> dst,
    Tensor<SrcT, Rank, MemorySpace::Shmem, SrcLayout> src,
    size_t worker_id, size_t worker_count) {
    size_t axes[Rank];
    if constexpr (is_same<DstLayout, StridedLayout<Rank>>::value)
        detail::stride_order(dst.layout, axes);
    else
        detail::row_major_order(axes);
    return detail::copy_ordered(dst, src, worker_id, worker_count, axes);
}

// A/B fragments: their native row-major layout is encoded in Fragment.
template <typename Fragment, typename RegisterLayoutT, typename T>
__device__ inline bool lds_to_fragment(
    Tensor<Fragment, 2, MemorySpace::Register, RegisterLayoutT>& dst,
    Tensor<T, 2, MemorySpace::Shmem, StridedLayout<2>> src) {
    if (!detail::valid_row_major_lds(dst.shape, src)) return false;
    dumma::ld_matrix(dst.fragment, src.data,
                     static_cast<unsigned>(src.layout.strides[0]));
    return true;
}

// Accumulator fragments: DUMMA requires an explicit memory layout.
template <typename Fragment, typename RegisterLayoutT, typename T,
          typename NativeLayout>
__device__ inline bool lds_to_fragment(
    Tensor<Fragment, 2, MemorySpace::Register, RegisterLayoutT>& dst,
    Tensor<T, 2, MemorySpace::Shmem, StridedLayout<2>> src,
    NativeLayout native_layout) {
    if (native_layout != du::dumma::mem_row_major ||
        !detail::valid_row_major_lds(dst.shape, src)) return false;
    dumma::ld_matrix(dst.fragment, src.data,
                     static_cast<unsigned>(src.layout.strides[0]), native_layout);
    return true;
}

// Store an accumulator fragment to row-major LDS. Every lane of the owning
// wave must call this; callers coordinate with other waves after the store.
template <typename T, typename Fragment, typename RegisterLayoutT>
__device__ inline bool fragment_to_lds(
    Tensor<T, 2, MemorySpace::Shmem, StridedLayout<2>> dst,
    const Tensor<Fragment, 2, MemorySpace::Register, RegisterLayoutT>& src) {
    static_assert(!is_const<T>::value, "LDS destination must be writable");
    if (!detail::valid_row_major_lds(src.shape, dst)) return false;
    dumma::store_matrix(dst.data, src.fragment,
                        static_cast<unsigned>(dst.layout.strides[0]),
                        du::dumma::mem_row_major);
    return true;
}


// ---------------------------------------------------------------------------
// Vectorized global <-> LDS I/O.
//
// A wave of 64 lanes issuing 16-byte loads/stores touches 1 KiB per
// instruction, and adjacent lanes cover adjacent 16-byte chunks. The helpers
// below choose that path only when the pointers and strides satisfy the vector
// alignment/width rules; otherwise they fall back to the same scalar loop, so
// callers can use one API for both fast and tail cases.
//
// The helpers operate on contiguous runs of elements. The row variants assume
// the caller has already checked layout.strides[1] == 1; they only vectorize
// when the row stride and base pointers keep every row 16-byte aligned.
//
// Typical use in an op:
//
//   copy::global_to_lds_vec<float4>(page, hbm + base, count, lane, wave);
//   copy::zero_lds_vec<float4>(page + count, tile - count, lane, wave);
//
// Always place the worker loop over vector indices, not element indices:
// lanes 0..63 then read/write 16 B each at consecutive addresses.

template <typename Vec>
struct VecTraits;

template <>
struct VecTraits<float4> {
    using Element = float;
    static constexpr size_t kLanes = 4;
    static DMK_HOST_DEVICE inline float4 splat(float value) {
        return float4{value, value, value, value};
    }
};

template <>
struct VecTraits<float2> {
    using Element = float;
    static constexpr size_t kLanes = 2;
    static DMK_HOST_DEVICE inline float2 splat(float value) {
        return float2{value, value};
    }
};

template <>
struct VecTraits<int4> {
    using Element = int;
    static constexpr size_t kLanes = 4;
    static DMK_HOST_DEVICE inline int4 splat(int value) {
        return int4{value, value, value, value};
    }
};

template <>
struct VecTraits<int2> {
    using Element = int;
    static constexpr size_t kLanes = 2;
    static DMK_HOST_DEVICE inline int2 splat(int value) {
        return int2{value, value};
    }
};

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool vec_aligned(const T* pointer) {
    return (reinterpret_cast<uintptr_t>(pointer) & (alignof(Vec) - 1)) == 0;
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline Vec load_vec(const T* pointer) {
    static_assert(sizeof(T) == sizeof(typename VecTraits<Vec>::Element),
                  "vector element type does not match T");
    return *reinterpret_cast<const Vec*>(pointer);
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline void store_vec(T* pointer, Vec value) {
    static_assert(sizeof(T) == sizeof(typename VecTraits<Vec>::Element),
                  "vector element type does not match T");
    *reinterpret_cast<Vec*>(pointer) = value;
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool can_vectorize(const T* dst, const T* src,
                                            size_t count) {
    return count >= VecTraits<Vec>::kLanes && dst != nullptr &&
           src != nullptr && vec_aligned<Vec>(dst) && vec_aligned<Vec>(src);
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool copy_vec(T* dst, const T* src, size_t count,
                                      size_t worker_id, size_t worker_count) {
    if (count == 0 || dst == nullptr || src == nullptr) return false;
    if (worker_count == 0 || worker_id >= worker_count) return false;

    constexpr size_t kLanes = VecTraits<Vec>::kLanes;
    if (can_vectorize<Vec>(dst, src, count)) {
        const size_t vectors = count / kLanes;
        for (size_t vector = worker_id; vector < vectors;
             vector += worker_count) {
            store_vec<Vec>(dst + vector * kLanes,
                           load_vec<Vec>(src + vector * kLanes));
        }
        const size_t tail_start = vectors * kLanes;
        for (size_t index = tail_start + worker_id; index < count;
             index += worker_count) {
            dst[index] = src[index];
        }
        return true;
    }

    for (size_t index = worker_id; index < count; index += worker_count)
        dst[index] = src[index];
    return false;
}

// HBM -> LDS, LDS -> HBM, LDS -> LDS. All three use the same aligned vector
// copy; the names document the intended memory spaces at the call site.
template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool global_to_lds_vec(T* dst, const T* src, size_t count,
                                               size_t worker_id,
                                               size_t worker_count) {
    return copy_vec<Vec>(dst, src, count, worker_id, worker_count);
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool lds_to_global_vec(T* dst, const T* src, size_t count,
                                               size_t worker_id,
                                               size_t worker_count) {
    return copy_vec<Vec>(dst, src, count, worker_id, worker_count);
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool lds_to_lds_vec(T* dst, const T* src, size_t count,
                                            size_t worker_id,
                                            size_t worker_count) {
    return copy_vec<Vec>(dst, src, count, worker_id, worker_count);
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool fill_lds_vec(T* dst, T value, size_t count,
                                          size_t worker_id,
                                          size_t worker_count) {
    if (count == 0 || dst == nullptr) return false;
    if (worker_count == 0 || worker_id >= worker_count) return false;

    constexpr size_t kLanes = VecTraits<Vec>::kLanes;
    // A tail starts at an arbitrary element offset (e.g. a partial tile), so
    // fill the unaligned head scalar, then vectorize the aligned middle.
    size_t head = 0;
    while (head < count && !vec_aligned<Vec>(dst + head)) ++head;
    const size_t vectors = (count - head) / kLanes;
    if (vectors != 0) {
        const Vec vector = VecTraits<Vec>::splat(value);
        for (size_t index = worker_id; index < head; index += worker_count)
            dst[index] = value;
        for (size_t vector_index = worker_id; vector_index < vectors;
             vector_index += worker_count) {
            store_vec<Vec>(dst + head + vector_index * kLanes, vector);
        }
        const size_t tail_start = head + vectors * kLanes;
        for (size_t index = tail_start + worker_id; index < count;
             index += worker_count) {
            dst[index] = value;
        }
        return true;
    }

    for (size_t index = worker_id; index < count; index += worker_count)
        dst[index] = value;
    return false;
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool zero_lds_vec(T* dst, size_t count,
                                          size_t worker_id,
                                          size_t worker_count) {
    return fill_lds_vec<Vec>(dst, T{}, count, worker_id, worker_count);
}

// Row-major copy between a strided source and a strided destination. The
// column dimension must be contiguous (stride 1); ops that can present a
// transposed view must use the scalar path instead. The vector fast path
// requires `cols`, both row strides and both base pointers to keep each row
// 16-byte aligned; otherwise the whole tile falls back to scalar.
template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool rows_copy_vec(T* dst, size_t dst_stride,
                                           const T* src, size_t src_stride,
                                           size_t rows, size_t cols,
                                           size_t worker_id,
                                           size_t worker_count) {
    if (rows == 0 || cols == 0 || dst == nullptr || src == nullptr)
        return false;
    if (worker_count == 0 || worker_id >= worker_count) return false;

    constexpr size_t kLanes = VecTraits<Vec>::kLanes;
    const bool vector_ok =
        cols % kLanes == 0 && src_stride % kLanes == 0 &&
        dst_stride % kLanes == 0 &&
        can_vectorize<Vec>(dst, src, rows * cols);
    if (vector_ok) {
        const size_t vectors_per_row = cols / kLanes;
        const size_t total = rows * vectors_per_row;
        for (size_t vector = worker_id; vector < total;
             vector += worker_count) {
            const size_t row = vector / vectors_per_row;
            const size_t col = (vector % vectors_per_row) * kLanes;
            store_vec<Vec>(dst + row * dst_stride + col,
                           load_vec<Vec>(src + row * src_stride + col));
        }
        return true;
    }

    const size_t total = rows * cols;
    for (size_t index = worker_id; index < total; index += worker_count) {
        const size_t row = index / cols;
        const size_t col = index % cols;
        dst[row * dst_stride + col] = src[row * src_stride + col];
    }
    return false;
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool global_rows_to_lds_vec(
    T* dst, size_t dst_stride, const T* src, size_t src_stride, size_t rows,
    size_t cols, size_t worker_id, size_t worker_count) {
    return rows_copy_vec<Vec>(dst, dst_stride, src, src_stride, rows, cols,
                              worker_id, worker_count);
}

template <typename Vec, typename T>
DMK_HOST_DEVICE inline bool lds_rows_to_global_vec(
    T* dst, size_t dst_stride, const T* src, size_t src_stride, size_t rows,
    size_t cols, size_t worker_id, size_t worker_count) {
    return rows_copy_vec<Vec>(dst, dst_stride, src, src_stride, rows, cols,
                              worker_id, worker_count);
}


}  // namespace dmk::copy
