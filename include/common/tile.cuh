#pragma once
#include "config.cuh"
#include <cstdint>
#include <type_traits>

namespace dcu {
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
    static constexpr int rows = Rows, cols = Cols, elements = Rows * Cols;
    T data[elements];
    DCU_HD T& operator()(int row, int col) { return data[row * Cols + col]; }
    DCU_HD const T& operator()(int row, int col) const { return data[row * Cols + col]; }
};

// Declare __shared__ shared_tile<...> inside a HIP kernel. No implicit barrier.
template<typename T, int Rows, int Cols>
using shared_tile = register_tile<T, Rows, Cols>;
} // namespace dcu
