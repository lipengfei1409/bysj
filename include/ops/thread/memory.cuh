#pragma once
#include "../../common/tile.cuh"

namespace dcu::thread {
template<typename T>
DCU_HD std::remove_const_t<T> load_masked(
    matrix_view<T> src, std::int64_t row, std::int64_t col) {
    return src.contains(row, col) ? src(row, col) : std::remove_const_t<T>{};
}
template<typename T>
DCU_HD void store_masked(matrix_view<T> dst, std::int64_t row,
                         std::int64_t col, T value) {
    static_assert(!std::is_const_v<T>, "store destination must be mutable");
    if (dst.contains(row, col)) dst(row, col) = value;
}
template<typename T, int R, int C>
DCU_HD void fill(register_tile<T, R, C>& dst, T value) {
    for (int i = 0; i < R * C; ++i) dst.data[i] = value;
}
template<typename T, typename U, int R, int C>
DCU_HD void load(register_tile<T, R, C>& dst, matrix_view<U> src,
                 std::int64_t row = 0, std::int64_t col = 0) {
    static_assert(std::is_same_v<T, std::remove_const_t<U>>, "load dtype mismatch");
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
        dst(r, c) = load_masked(src, row + r, col + c);
}
template<typename T, int R, int C>
DCU_HD void store(matrix_view<T> dst, const register_tile<T, R, C>& src,
                  std::int64_t row = 0, std::int64_t col = 0) {
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
        store_masked(dst, row + r, col + c, src(r, c));
}
} // namespace dcu::thread
