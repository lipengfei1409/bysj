#pragma once

#include <hip/hip_runtime.h>
#include <du_mma.h>

#include "layout.h"

namespace dmk::copy {

// DUMMA matrix loads/stores are collective operations of one complete wave.
// Operand layout is encoded in the fragment type; accumulator memory layout
// is passed explicitly to the overloads below.
template <typename Fragment, typename T>
__device__ inline void ld_matrix(Fragment& fragment, const T* src, unsigned ld) {
    ::du::dumma::du_load_matrix_sync(fragment, src, ld);
}

template <typename Fragment, typename T, typename NativeLayout>
__device__ inline void ld_matrix(Fragment& fragment, const T* src, unsigned ld,
                                 NativeLayout layout) {
    ::du::dumma::du_load_matrix_sync(fragment, src, ld, layout);
}

template <typename T, typename Fragment, typename NativeLayout>
__device__ inline void store_matrix(T* dst, const Fragment& fragment,
                                    unsigned ld, NativeLayout layout) {
    ::du::dumma::du_store_matrix_sync(dst, fragment, ld, layout);
}

// Masked HBM matrix -> LDS tile. The caller supplies a physical LDS layout
// and enough storage for its leading dimension. DUMMA reads row-major tiles.
template <int Rows, int Cols, typename T>
__device__ inline void stage_to_shmem(T* tile, int tile_ld,
                                      ShmemLayout tile_layout,
                                      const T* src, int src_rows, int src_cols,
                                      int src_ld, HbmLayout src_layout,
                                      int row_start, int col_start) {
    const int lane = threadIdx.x % warpSize;
    for (int index = lane; index < Rows * Cols; index += warpSize) {
        const int local_row = index / Cols;
        const int local_col = index % Cols;
        const int row = row_start + local_row;
        const int col = col_start + local_col;
        tile[matrix_offset(tile_layout, local_row, local_col, tile_ld)] =
            row < src_rows && col < src_cols
                ? src[matrix_offset(src_layout, row, col, src_ld)] : T(0);
    }
}

// Masked LDS tile -> HBM matrix.
template <int Rows, int Cols, typename T>
__device__ inline void store_from_shmem(T* dst, int dst_rows, int dst_cols,
                                        int dst_ld, HbmLayout dst_layout,
                                        const T* tile, int tile_ld,
                                        ShmemLayout tile_layout,
                                        int row_start, int col_start) {
    const int lane = threadIdx.x % warpSize;
    for (int index = lane; index < Rows * Cols; index += warpSize) {
        const int local_row = index / Cols;
        const int local_col = index % Cols;
        const int row = row_start + local_row;
        const int col = col_start + local_col;
        if (row < dst_rows && col < dst_cols)
            dst[matrix_offset(dst_layout, row, col, dst_ld)] =
                tile[matrix_offset(tile_layout, local_row, local_col, tile_ld)];
    }
}

}  // namespace dmk::copy
