#pragma once
#include "../../common/tile.cuh"
#include "../thread/memory.cuh"
#include "wave.cuh"
#if defined(__HIPCC__)
namespace dcu::group {
// Whole-block cooperative synchronous copy. Caller inserts block_sync before
// consuming LDS and again before reusing a double-buffer slot. Not TMA/cp.async.
template<typename T, typename U, int R, int C>
DCU_DEVICE void load(shared_tile<T, R, C>& dst, matrix_view<U> src,
                     std::int64_t row = 0, std::int64_t col = 0) {
    static_assert(std::is_same_v<T, std::remove_const_t<U>>, "load dtype mismatch");
    for (unsigned i = thread_rank(); i < R * C; i += block_threads())
        dst.data[i] = thread::load_masked(src, row + i / C, col + i % C);
}
template<typename T, int R, int C>
DCU_DEVICE void store(matrix_view<T> dst, const shared_tile<T, R, C>& src,
                      std::int64_t row = 0, std::int64_t col = 0) {
    for (unsigned i = thread_rank(); i < R * C; i += block_threads())
        thread::store_masked(dst, row + i / C, col + i % C, src.data[i]);
}
} // namespace dcu::group
#endif
