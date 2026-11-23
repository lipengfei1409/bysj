#pragma once

// Tile views: a tile is a block of a tensor (tile_m x tile_n, or tile_b x
// tile_m x tile_n for a batched rank-3 view), not a contiguous run of storage.
// The parent tensor keeps its ordinary strided view; a tile view folds the
// tile origin into the pointer and keeps the parent's strides, so an operator
// can describe its access pattern with one tile shape and fetch a whole tile
// with one call.
//
// Each operator picks its own tile extents (GEMM: 16 x K for A, K x 16 for B,
// 16 x 16 for C; attention: 16 x D for Q/K/V/O) and uses
//
//   auto src = tile::make_tile(tensor, tile_shape, {m_tile, n_tile});
//   auto dst = tile::make_lds_view<T, 2>(page, {16, k}, strides);
//   tile::load_tile(dst, src, lane, wave_size);   // zero + clamped copy
//
// Tiles at the tensor edge are clamped: make_tile shrinks the logical shape,
// so a copy never touches memory outside the tensor, and load_tile zero-fills
// the rest of the destination tile.
//
// The copies keep the coalescing rules of the operators:
//   - the fast path moves 16 B chunks along the innermost axis when both views
//     are contiguous there, every other stride is a multiple of the vector
//     width and the pointers are aligned;
//   - the scalar fallback walks the tile in increasing-stride order, so
//     neighbouring workers touch neighbouring addresses for row-major,
//     column-major and rank-3 views alike.

#include "copy.h"
#include "tensor.h"

#include <cstddef>
#include <type_traits>

namespace dmk::tile {

using tensor::MemorySpace;
using tensor::Shape;
using tensor::StridedLayout;
using tensor::Tensor;

template <size_t Rank>
struct TileShape {
    size_t dims[Rank]{};
};

// One tile of a tensor: origin folded into `data`, logical shape clamped at
// the tensor edge, parent strides preserved.
template <typename T, size_t Rank, MemorySpace Space>
struct TileView {
    using element = T;
    static constexpr size_t rank = Rank;

    T* data = nullptr;
    Shape<Rank> shape{};
    StridedLayout<Rank> layout{};

    DMK_HOST_DEVICE constexpr size_t numel() const { return shape.numel(); }

    DMK_HOST_DEVICE constexpr size_t offset(const Shape<Rank>& index) const {
        return layout.offset(shape, index);
    }

    DMK_HOST_DEVICE constexpr bool is_contiguous() const {
        return layout.is_contiguous(shape);
    }
};

// Number of tiles that cover `shape` along `axis`.
template <size_t Rank>
DMK_HOST_DEVICE constexpr size_t tile_count(const Shape<Rank>& shape,
                                            const TileShape<Rank>& tile,
                                            size_t axis) {
    return tile.dims[axis] == 0
               ? 0
               : (shape.dims[axis] + tile.dims[axis] - 1) / tile.dims[axis];
}

// A dense LDS tile (row-major) or one with explicit strides, e.g. the padded
// attention pages.
template <typename T, size_t Rank>
DMK_HOST_DEVICE inline TileView<T, Rank, MemorySpace::Shmem> make_lds_view(
    T* data, const Shape<Rank>& shape) {
    TileView<T, Rank, MemorySpace::Shmem> view;
    view.data = data;
    view.shape = shape;
    view.layout = StridedLayout<Rank>::contiguous(shape);
    return view;
}

template <typename T, size_t Rank>
DMK_HOST_DEVICE inline TileView<T, Rank, MemorySpace::Shmem> make_lds_view(
    T* data, const Shape<Rank>& shape, const StridedLayout<Rank>& layout) {
    TileView<T, Rank, MemorySpace::Shmem> view;
    view.data = data;
    view.shape = shape;
    view.layout = layout;
    return view;
}

// The tile at `coord` of a tensor tiled with `tile`. The result is clamped to
// the tensor edge and is empty when the coordinate is past the end.
template <typename T, size_t Rank, MemorySpace Space, typename Layout>
DMK_HOST_DEVICE inline TileView<T, Rank, Space> make_tile(
    const Tensor<T, Rank, Space, Layout>& tensor, const TileShape<Rank>& tile,
    const size_t (&coord)[Rank]) {
    Shape<Rank> origin{};
    bool inside = true;
    for (size_t axis = 0; axis < Rank; ++axis) {
        origin.dims[axis] = coord[axis] * tile.dims[axis];
        if (origin.dims[axis] >= tensor.shape.dims[axis]) inside = false;
    }

    TileView<T, Rank, Space> view;
    if (!inside) return view;  // empty: numel() == 0, no copy happens
    view.data = tensor.data + tensor.offset(origin);
    for (size_t axis = 0; axis < Rank; ++axis) {
        const size_t remaining = tensor.shape.dims[axis] - origin.dims[axis];
        view.shape.dims[axis] =
            tile.dims[axis] < remaining ? tile.dims[axis] : remaining;
        view.layout.strides[axis] = tensor.layout.strides[axis];
    }
    return view;
}

// The tile of `tile` extents whose top-left corner is at `origin` (in tensor
// coordinates), clamped to the tensor edge. Unlike make_tile this does not
// require a tile-aligned origin, which is what instruction words carry: e.g.
// a GEMM op asks for the 16x16 block at (m0, n0).
template <typename T, size_t Rank, MemorySpace Space, typename Layout>
DMK_HOST_DEVICE inline TileView<T, Rank, Space> make_tile_at(
    const Tensor<T, Rank, Space, Layout>& tensor, const TileShape<Rank>& tile,
    const size_t (&origin)[Rank]) {
    TileView<T, Rank, Space> view;
    bool inside = true;
    Shape<Rank> origin_index{};
    for (size_t axis = 0; axis < Rank; ++axis) {
        origin_index.dims[axis] = origin[axis];
        if (origin[axis] >= tensor.shape.dims[axis]) inside = false;
    }
    if (!inside) return view;
    view.data = tensor.data + tensor.offset(origin_index);
    for (size_t axis = 0; axis < Rank; ++axis) {
        const size_t remaining = tensor.shape.dims[axis] - origin[axis];
        view.shape.dims[axis] =
            tile.dims[axis] < remaining ? tile.dims[axis] : remaining;
        view.layout.strides[axis] = tensor.layout.strides[axis];
    }
    return view;
}

// Offset of the `linear`-th element of a view in row-major order.
template <typename View>
DMK_HOST_DEVICE inline size_t row_major_offset(const View& view,
                                               size_t linear) {
    constexpr size_t Rank = View::rank;
    Shape<Rank> index{};
    for (size_t axis = 0; axis < Rank; ++axis) {
        const size_t extent = view.shape.dims[axis];
        index.dims[axis] = extent != 0 ? linear % extent : 0;
        linear = extent != 0 ? linear / extent : 0;
    }
    return view.offset(index);
}

// Fill every element of a view.
template <typename View, typename T>
DMK_HOST_DEVICE inline void fill_tile(const View& view, T value,
                                      size_t worker_id,
                                      size_t worker_count) {
    if (worker_count == 0 || worker_id >= worker_count) return;
    const size_t count = view.numel();
    for (size_t linear = worker_id; linear < count; linear += worker_count)
        view.data[row_major_offset(view, linear)] = value;
}

// Copy one tile between two views of the same shape. Returns false when the
// views are inconsistent (shape mismatch, null data) or the worker is idle;
// the caller can use the return value to fall back to a masked path.
template <typename Dst, typename Src>
DMK_HOST_DEVICE inline bool copy_tile(const Dst& dst, const Src& src,
                                      size_t worker_id, size_t worker_count) {
    static_assert(Dst::rank == Src::rank,
                  "tile views must have the same rank");
    using T = typename Dst::element;
    static_assert(
        std::is_same<typename std::remove_const<typename Src::element>::type,
                     typename std::remove_const<T>::type>::value,
        "tile views must have the same element type");
    constexpr size_t Rank = Dst::rank;

    if (worker_count == 0 || worker_id >= worker_count) return false;
    if (dst.data == nullptr || src.data == nullptr) return false;
    for (size_t axis = 0; axis < Rank; ++axis)
        if (dst.shape.dims[axis] != src.shape.dims[axis]) return false;

    const size_t count = dst.numel();
    if (count == 0) return true;

    // 16 B chunks along the innermost axis, one row per worker step.
    constexpr size_t kVector = 16 / sizeof(typename Dst::element);
    const size_t inner = dst.shape.dims[Rank - 1];
    bool vector_ok = inner % kVector == 0 && inner != 0 &&
                     dst.layout.strides[Rank - 1] == 1 &&
                     src.layout.strides[Rank - 1] == 1 &&
                     copy::aligned<float4>(dst.data) &&
                     copy::aligned<float4>(src.data);
    for (size_t axis = 0; axis + 1 < Rank && vector_ok; ++axis) {
        if (dst.layout.strides[axis] % kVector != 0 ||
            src.layout.strides[axis] % kVector != 0)
            vector_ok = false;
    }
    if (vector_ok) {
        Shape<Rank> index{};
        const size_t rows = count / inner;
        for (size_t row = worker_id; row < rows; row += worker_count) {
            size_t remaining = row;
            for (size_t axis = Rank - 1; axis-- > 0;) {
                const size_t extent = dst.shape.dims[axis];
                index.dims[axis] = remaining % extent;
                remaining /= extent;
            }
            const size_t dst_base = dst.offset(index);
            const size_t src_base = src.offset(index);
            for (size_t col = 0; col < inner; col += kVector) {
                copy::store_vec<float4>(
                    dst.data + dst_base + col,
                    copy::load_vec<float4>(src.data + src_base + col));
            }
        }
        return true;
    }

    // Increasing-stride order keeps neighbouring workers on neighbouring
    // addresses for row-major and column-major views alike.
    size_t axes[Rank];
    for (size_t axis = 0; axis < Rank; ++axis) axes[axis] = axis;
    for (size_t position = 1; position < Rank; ++position) {
        const size_t axis = axes[position];
        size_t slot = position;
        while (slot > 0 && dst.layout.strides[axes[slot - 1]] >
                               dst.layout.strides[axis]) {
            axes[slot] = axes[slot - 1];
            --slot;
        }
        axes[slot] = axis;
    }
    for (size_t linear = worker_id; linear < count; linear += worker_count) {
        Shape<Rank> index{};
        size_t remaining = linear;
        for (size_t position = 0; position < Rank; ++position) {
            const size_t axis = axes[position];
            const size_t extent = dst.shape.dims[axis];
            index.dims[axis] = remaining % extent;
            remaining /= extent;
        }
        dst.data[dst.offset(index)] = src.data[src.offset(index)];
    }
    return true;
}

// Load one tile: a full tile is a plain (vectorized) copy; a tile clamped at
// the tensor edge is copied in a single pass that writes zeros outside the
// source shape, so no worker ever zeroes another worker's data. The source
// must not be larger than the destination, which is what make_tile returns.
template <typename Dst, typename Src>
DMK_HOST_DEVICE inline bool load_tile(const Dst& dst, const Src& src,
                                      size_t worker_id, size_t worker_count) {
    static_assert(Dst::rank == Src::rank,
                  "tile views must have the same rank");
    if (worker_count == 0 || worker_id >= worker_count) return false;
    constexpr size_t Rank = Dst::rank;
    for (size_t axis = 0; axis < Rank; ++axis)
        if (src.shape.dims[axis] > dst.shape.dims[axis]) return false;

    bool full = true;
    for (size_t axis = 0; axis < Rank; ++axis)
        full = full && src.shape.dims[axis] == dst.shape.dims[axis];
    if (full) return copy_tile(dst, src, worker_id, worker_count);

    // Edge tile: one pass over the destination so neighbours never overwrite
    // each other's fill.
    if (dst.data == nullptr) return false;
    const size_t count = dst.numel();
    for (size_t linear = worker_id; linear < count; linear += worker_count) {
        Shape<Rank> index{};
        size_t remaining = linear;
        for (size_t axis = Rank; axis-- > 0;) {
            index.dims[axis] = remaining % dst.shape.dims[axis];
            remaining /= dst.shape.dims[axis];
        }
        bool inside = src.data != nullptr;
        for (size_t axis = 0; axis < Rank; ++axis)
            inside = inside && index.dims[axis] < src.shape.dims[axis];
        dst.data[dst.offset(index)] =
            inside ? src.data[src.offset(index)] : typename Dst::element{};
    }
    return true;
}

}  // namespace dmk::tile
