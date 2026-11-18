#pragma once

#include "tensor.h"
#include "dumma.h"

namespace dmk::copy {

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
    ld_matrix(dst.fragment, src.data, static_cast<unsigned>(src.layout.strides[0]));
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
    ld_matrix(dst.fragment, src.data,
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
    store_matrix(dst.data, src.fragment, static_cast<unsigned>(dst.layout.strides[0]),
                 du::dumma::mem_row_major);
    return true;
}

}  // namespace dmk::copy
