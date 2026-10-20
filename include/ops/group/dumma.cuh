#pragma once
#include "../../common/config.cuh"

// Direct facade for DTK 26.04 DUMMA manual, section 2 and section 4 example.
// No PTX, software matmul fallback, pointer copying or dtype conversion here.
#if DPA4_ENABLE_DUMMA
#if !defined(__HIPCC__)
#error "DPA4_ENABLE_DUMMA requires the DTK HIP compiler"
#elif !__has_include(<du_mma.h>)
#error "DUMMA requested but du_mma.h is unavailable; use a compatible DTK installation"
#else
#include <hip/hip_fp16.h>
#include <du_mma.h>
#define DCU_DUMMA_AVAILABLE 1

namespace dcu::dumma {
namespace native = ::du::dumma;
// Preserve the native fragment type/ABI and the documented role/layout tags.
using native::DUFragment;
using native::matrix_a;
using native::matrix_b;
using native::accumulator;
using native::row_major;
using memory_layout = decltype(native::mem_row_major);
inline constexpr memory_layout row_major_memory = native::mem_row_major;

// Entire, converged wave64 must execute each call with uniform arguments.
// Native DTK controls fragment distribution, supported shapes and types.
template<typename Fragment, typename T>
DCU_DEVICE void fill_fragment(Fragment& dst, const T& value) {
    native::du_fill_fragment(dst, value);
}

// A/B memory layout is encoded in the fragment type; ld is in elements.
template<typename Fragment, typename T>
DCU_DEVICE void load_matrix_sync(Fragment& dst, const T* src, unsigned ld) {
    native::du_load_matrix_sync(dst, src, ld);
}

// C accumulator uses the runtime memory layout overload from manual 2.2.
template<typename Fragment, typename T>
DCU_DEVICE void load_matrix_sync(Fragment& dst, const T* src, unsigned ld,
                                 memory_layout layout) {
    native::du_load_matrix_sync(dst, src, ld, layout);
}

template<typename D, typename A, typename B, typename C>
DCU_DEVICE void mma_sync(D& d, const A& a, const B& b, const C& c) {
    native::du_mma_sync(d, a, b, c);
}

// dst is device-accessible memory. Host readback is a separate hipMemcpy.
template<typename T, typename Fragment>
DCU_DEVICE void store_matrix_sync(T* dst, const Fragment& src, unsigned ld,
                                  memory_layout layout = row_major_memory) {
    native::du_store_matrix_sync(dst, src, ld, layout);
}
} // namespace dcu::dumma
#endif // compiler/header available
#endif // DPA4_ENABLE_DUMMA
