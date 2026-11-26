#pragma once

#include "ops/utils.h"

namespace dmk::dumma {

// Native DUMMA types and layout tags remain available through native::.
namespace native = ::du::dumma;

// All lanes in the owning wave call these functions with the same arguments.
// The native DUMMA API checks whether the fragment type and tile shape are
// supported by the selected DCU target.
template <typename Fragment, typename T>
__device__ inline void fill_fragment(Fragment& dst, const T& value) {
    native::du_fill_fragment(dst, value);
}

// Matrix A/B: their memory layout is encoded in the fragment type.
template <typename Fragment, typename T>
__device__ inline void ld_matrix(Fragment& dst, const T* src, unsigned ld) {
    native::du_load_matrix_sync(dst, src, ld);
}

// Accumulator matrix C: the memory layout is passed explicitly.
template <typename Fragment, typename T, typename NativeLayout>
__device__ inline void ld_matrix(Fragment& dst, const T* src, unsigned ld,
                                 NativeLayout layout) {
    native::du_load_matrix_sync(dst, src, ld, layout);
}

template <typename DstFragment, typename AFragment, typename BFragment,
          typename CFragment>
__device__ inline void mma(DstFragment& dst, const AFragment& a,
                           const BFragment& b, const CFragment& c) {
    native::du_mma_sync(dst, a, b, c);
}

// Store an accumulator fragment; ld is the destination's leading dimension
// in elements, and layout describes that destination buffer.
template <typename T, typename Fragment, typename NativeLayout>
__device__ inline void store_matrix(T* dst, const Fragment& src, unsigned ld,
                                    NativeLayout layout) {
    native::du_store_matrix_sync(dst, src, ld, layout);
}

}  // namespace dmk::dumma
