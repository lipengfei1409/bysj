#pragma once
#include "../../common/config.h"

// Thin facade over the DTK 26.04 DUMMA API. The native fragments remain
// opaque; this header does not emulate the instruction or provide a fallback.
#if DPA4_ENABLE_DUMMA
#if !defined(__HIPCC__)
#error "DPA4_ENABLE_DUMMA requires the DTK HIP compiler"
#elif !__has_include(<du_mma.h>)
#error "DUMMA requested but du_mma.h is unavailable; use a compatible DTK installation"
#else
#include <du_mma.h>
#define DCU_DUMMA_AVAILABLE 1

namespace dmk::dumma {
namespace native = ::du::dumma;
using native::DUFragment;
using native::matrix_a;
using native::matrix_b;
using native::accumulator;
using native::row_major;
using layout_t = decltype(native::mem_row_major);
using memory_layout = layout_t;
inline constexpr layout_t row_major_memory = native::mem_row_major;

// A complete, converged wave64 must execute each call with uniform arguments.
// A/B layout is encoded in the fragment type; ldm is measured in elements.
template<typename Fragment, typename T>
DCU_DEVICE void du_fill_fragment(Fragment& dst, const T& value) {
    native::du_fill_fragment(dst, value);
}

template<typename Fragment, typename T>
DCU_DEVICE void du_load_matrix_sync(Fragment& dst, const T* src, unsigned ldm) {
    native::du_load_matrix_sync(dst, src, ldm);
}

template<typename Fragment, typename T>
DCU_DEVICE void du_load_matrix_sync(Fragment& dst, const T* src, unsigned ldm,
                                   layout_t layout) {
    native::du_load_matrix_sync(dst, src, ldm, layout);
}

template<typename D, typename A, typename B, typename C>
DCU_DEVICE void du_mma_sync(D& d, const A& a, const B& b, const C& c) {
    native::du_mma_sync(d, a, b, c);
}

// mptr is device-accessible memory; host readback is a separate hipMemcpy.
template<typename T, typename Fragment>
DCU_DEVICE void du_store_matrix_sync(T* mptr, const Fragment& src,
                                     unsigned ldm, layout_t layout) {
    native::du_store_matrix_sync(mptr, src, ldm, layout);
}
} // namespace dmk::dumma
#endif // compiler/header available
#endif // DPA4_ENABLE_DUMMA
