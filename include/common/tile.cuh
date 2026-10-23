#pragma once
#include "config.cuh"
#include <cstdint>
#include <type_traits>

namespace dcu {
enum class tile_order { row_major };

// Compile-time layout metadata; it does not describe DUMMA's opaque lane mapping.
template<int Rows, int Cols, tile_order Order = tile_order::row_major>
struct tile_layout {
    static_assert(Rows > 0 && Cols > 0, "tile dimensions must be positive");
    static_assert(Order == tile_order::row_major, "unsupported tile layout");
    static constexpr int rows = Rows;
    static constexpr int cols = Cols;
    static constexpr int elements = Rows * Cols;
    DCU_HD static constexpr int offset(int row, int col) {
        return row * Cols + col;
    }
};

template<int M, int N, int K>
struct gemm_tile_shape {
    static_assert(M > 0 && N > 0 && K > 0, "GEMM tile dimensions must be positive");
    static constexpr int m = M;
    static constexpr int n = N;
    static constexpr int k = K;
    using a_layout = tile_layout<M, K>;
    using b_layout = tile_layout<K, N>;
    using c_layout = tile_layout<M, N>;
};

// Row-major, non-owning view. stride is in ELEMENTS, not bytes.
// Caller validates once before launch. masked operations only check coordinates.
template<typename T>
struct matrix_view {
    T* data;
    std::int64_t rows, cols, stride;

    DCU_HD bool valid() const {
        return rows >= 0 && cols >= 0 && stride >= cols &&
               ((rows == 0 || cols == 0) || data != nullptr);
    }
    DCU_HD bool contains(std::int64_t row, std::int64_t col) const {
        return row >= 0 && row < rows && col >= 0 && col < cols;
    }
    DCU_HD T& operator()(std::int64_t row, std::int64_t col) const {
        return data[row * stride + col];
    }
};

// Whole tile owned by ONE thread, not a distributed DUMMA fragment.
// Intended for tiny tensor operations/reference tests; large tiles can spill.
template<typename T, int Rows, int Cols>
struct alignas(16) register_tile {
    static_assert(Rows > 0 && Cols > 0, "tile dimensions must be positive");
    static_assert(!std::is_const_v<T>, "register tile must be mutable");
    using value_type = T;
    using layout = tile_layout<Rows, Cols>;
    static constexpr int rows = layout::rows, cols = layout::cols;
    static constexpr int elements = layout::elements;
    T data[elements];
    DCU_HD T& operator()(int row, int col) { return data[layout::offset(row, col)]; }
    DCU_HD const T& operator()(int row, int col) const { return data[layout::offset(row, col)]; }
};

// Declare __shared__ shared_tile<...> inside a HIP kernel. No implicit barrier.
template<typename T, int Rows, int Cols>
using shared_tile = register_tile<T, Rows, Cols>;
} // namespace dcu
