#pragma once

#include "utils.h"

namespace dmk::tensor {
enum class MemorySpace { HBM, Shmem, Register };

// Fixed rank, runtime extents. No heap allocation is needed in device code.
template <size_t Rank>
struct Shape {
    static_assert(Rank > 0, "Shape rank must be positive");
    size_t dims[Rank]{};

    DMK_HOST_DEVICE constexpr size_t ndim() const { return Rank; }

    DMK_HOST_DEVICE constexpr bool try_numel(size_t& result) const {
        for (size_t axis = 0; axis < Rank; ++axis) {
            if (dims[axis] == 0) {
                result = 0;
                return true;
            }
        }
        result = 1;
        for (size_t axis = 0; axis < Rank; ++axis) {
            if (result > ~size_t{0} / dims[axis]) return false;
            result *= dims[axis];
        }
        return true;
    }

    DMK_HOST_DEVICE constexpr size_t numel() const {
        size_t result = 0;
        const bool valid = try_numel(result);
        assert(valid && "Shape element count overflow");
        return valid ? result : 0;
    }
};

// Element strides for HBM or ordinary LDS tiles.
template <size_t Rank>
struct StridedLayout {
    size_t strides[Rank]{};

    DMK_HOST_DEVICE static constexpr StridedLayout contiguous(const Shape<Rank>& shape) {
        StridedLayout result{};
        size_t step = 1;
        for (size_t axis = Rank; axis-- > 0;) {
            result.strides[axis] = step;
            step *= shape.dims[axis];
        }
        return result;
    }

    DMK_HOST_DEVICE constexpr size_t offset(const Shape<Rank>&, const Shape<Rank>& indices) const {
        size_t result = 0;
        for (size_t axis = 0; axis < Rank; ++axis)
            result += indices.dims[axis] * strides[axis];
        return result;
    }

    DMK_HOST_DEVICE constexpr bool is_contiguous(const Shape<Rank>& shape) const {
        for (size_t axis = 0; axis < Rank; ++axis)
            if (shape.dims[axis] == 0) return true;
        size_t step = 1;
        for (size_t axis = Rank; axis-- > 0;) {
            if (shape.dims[axis] != 1 && strides[axis] != step) return false;
            step *= shape.dims[axis];
        }
        return true;
    }
};

template <size_t Rank>
using HbmLayout = StridedLayout<Rank>;

template <size_t Rank>
using ShmemStridedLayout = StridedLayout<Rank>;

// Two-dimensional LDS layout. Each row permutes 32 bank words by XOR with its
// row index. Padding makes the permutation stay within the row's allocation.
// ElementBytes must divide the 4-byte bank width (FP32, FP16, BF16, INT8).
template <size_t ElementBytes>
struct ShmemSwizzleLayout {
    static_assert(ElementBytes == 1 || ElementBytes == 2 || ElementBytes == 4,
                  "Swizzle supports 1-, 2-, or 4-byte elements");
    static constexpr size_t element_bytes = ElementBytes;
    static constexpr size_t bank_count = 32;
    static constexpr size_t bank_width_bytes = 4;
    static constexpr size_t elements_per_bank = bank_width_bytes / ElementBytes;
    static constexpr size_t group_elements = bank_count * elements_per_bank;

    uint32_t row_xor_mask = bank_count - 1;

    DMK_HOST_DEVICE constexpr size_t row_stride(const Shape<2>& shape) const {
        const size_t cols = shape.dims[1];
        return (cols / group_elements + (cols % group_elements != 0)) * group_elements;
    }

    DMK_HOST_DEVICE constexpr size_t required_elements(const Shape<2>& shape) const {
        return shape.dims[0] * row_stride(shape);
    }

    DMK_HOST_DEVICE constexpr bool try_required_elements(const Shape<2>& shape,
                                                        size_t& result) const {
        const size_t groups = shape.dims[1] / group_elements +
                                   (shape.dims[1] % group_elements != 0);
        if (groups > ~size_t{0} / group_elements) return false;
        const size_t stride = groups * group_elements;
        if (stride && shape.dims[0] > ~size_t{0} / stride) return false;
        result = shape.dims[0] * stride;
        return true;
    }

    DMK_HOST_DEVICE constexpr size_t offset(const Shape<2>& shape,
                                                const Shape<2>& indices) const {
        const size_t row = indices.dims[0];
        const size_t col = indices.dims[1];
        const size_t word = col / elements_per_bank;
        const size_t lane = col % elements_per_bank;
        const size_t physical_word = (word & ~(bank_count - 1)) |
            ((word & (bank_count - 1)) ^
             (row & row_xor_mask & (bank_count - 1)));
        return row * row_stride(shape) + physical_word * elements_per_bank + lane;
    }

    DMK_HOST_DEVICE constexpr bool is_contiguous(const Shape<2>&) const { return false; }
};

template <typename Layout>
struct IsShmemSwizzle : false_type {};

template <size_t ElementBytes>
struct IsShmemSwizzle<ShmemSwizzleLayout<ElementBytes>> : true_type {};

template <typename Layout, typename T>
struct SwizzleElementMatches : true_type {};

template <size_t ElementBytes, typename T>
struct SwizzleElementMatches<ShmemSwizzleLayout<ElementBytes>, T>
    : integral_constant<bool, sizeof(T) == ElementBytes> {};

// Fragment distribution is defined by the compute instruction, e.g. DUMMA.
struct RegisterLayout {};

template <size_t Rank, MemorySpace Space>
struct DefaultTensorLayout { using type = StridedLayout<Rank>; };

template <size_t Rank>
struct DefaultTensorLayout<Rank, MemorySpace::Register> {
    using type = RegisterLayout;
};

// A non-owning view. The caller manages the lifetime of HBM/LDS storage.
template <typename T, size_t Rank, MemorySpace Space, typename Layout = typename DefaultTensorLayout<Rank, Space>::type>
struct Tensor {
    static_assert(Space != MemorySpace::Register,
                  "Register Tensor uses its fragment specialization");
    static_assert(Space == MemorySpace::Shmem || !IsShmemSwizzle<Layout>::value,
                  "Swizzle layout is only valid in LDS");
    static_assert(SwizzleElementMatches<Layout, T>::value,
                  "Swizzle element width must match sizeof(T)");

    T* data = nullptr;
    Shape<Rank> shape{};
    Layout layout{};

    DMK_HOST_DEVICE constexpr size_t numel() const { return shape.numel(); }

    DMK_HOST_DEVICE constexpr size_t offset(const Shape<Rank>& indices) const {
        return layout.offset(shape, indices);
    }

    DMK_HOST_DEVICE constexpr bool is_contiguous() const {
        return layout.is_contiguous(shape);
    }

    template <size_t NewRank>
    DMK_HOST_DEVICE constexpr bool reshape(
        const Shape<NewRank>& new_shape,
        Tensor<T, NewRank, Space, StridedLayout<NewRank>>& result) const {
        if constexpr (!is_same<Layout, StridedLayout<Rank>>::value) {
            return false;
        } else {
            size_t old_count = 0;
            size_t new_count = 0;
            if (!is_contiguous() || !shape.try_numel(old_count) ||
                !new_shape.try_numel(new_count) || old_count != new_count)
                return false;
            result = {data, new_shape, StridedLayout<NewRank>::contiguous(new_shape)};
            return true;
        }
    }

    DMK_HOST_DEVICE constexpr Tensor transpose(size_t first,
                                             size_t second) const {
        static_assert(is_same<Layout, StridedLayout<Rank>>::value,
                      "Transpose of swizzled LDS requires a new physical layout");
        assert(first < Rank && second < Rank);
        Tensor result = *this;
        const size_t dim = result.shape.dims[first];
        result.shape.dims[first] = result.shape.dims[second];
        result.shape.dims[second] = dim;
        const size_t stride = result.layout.strides[first];
        result.layout.strides[first] = result.layout.strides[second];
        result.layout.strides[second] = stride;
        return result;
    }
};

// Register fragments are values distributed across lanes, not addressable T*.
template <typename Fragment, size_t Rank, typename Layout>
struct Tensor<Fragment, Rank, MemorySpace::Register, Layout> {
    Fragment fragment;
    Shape<Rank> shape;
    Layout layout;

    DMK_HOST_DEVICE constexpr size_t numel() const { return shape.numel(); }
};


// Host-only, zero-copy mapping from a Torch CUDA/DTK tensor. The Torch tensor
// owns the storage and must outlive every kernel using the returned view.
#ifndef __HIP_DEVICE_COMPILE__
template <typename T, size_t Rank, MemorySpace Space,
          typename Layout = typename DefaultTensorLayout<Rank, Space>::type>
inline void from_torch(at::Tensor& src, Tensor<T, Rank, Space, Layout>& out) {
    static_assert(Space == MemorySpace::HBM, "Torch data maps only to HBM");
    static_assert(is_same<Layout, StridedLayout<Rank>>::value,
                  "Torch data requires a strided HBM layout");
    using Element = remove_const_t<T>;
    TORCH_CHECK(src.defined(), "from_torch requires a defined tensor");
    TORCH_CHECK(src.is_cuda(), "from_torch requires a DCU/CUDA tensor");
    TORCH_CHECK(src.layout() == at::kStrided,
                "from_torch requires a strided tensor");
    TORCH_CHECK(src.dim() == static_cast<int64_t>(Rank),
                "from_torch rank mismatch");
    TORCH_CHECK(src.scalar_type() == c10::CppTypeToScalarType<Element>::value,
                "from_torch dtype mismatch");

    for (size_t axis = 0; axis < Rank; ++axis) {
        const int64_t dimension = src.size(static_cast<int64_t>(axis));
        const int64_t stride = src.stride(static_cast<int64_t>(axis));
        TORCH_CHECK(dimension >= 0 && stride >= 0,
                    "from_torch requires nonnegative dimensions and strides");
        out.shape.dims[axis] = static_cast<size_t>(dimension);
        out.layout.strides[axis] = static_cast<size_t>(stride);
    }
    out.data = src.data_ptr<Element>();
}

// Reinterpret the same Torch-owned storage with the DMK view's shape/strides.
// `out` must already own the same starting pointer; as_strided keeps that
// storage alive and allocates no new HBM buffer.
template <typename T, size_t Rank, MemorySpace Space,
          typename Layout = typename DefaultTensorLayout<Rank, Space>::type>
inline void to_torch(const Tensor<T, Rank, Space, Layout>& src,
                     at::Tensor& out) {
    static_assert(Space == MemorySpace::HBM, "Torch data maps only to HBM");
    static_assert(is_same<Layout, StridedLayout<Rank>>::value,
                  "Torch data requires a strided HBM layout");
    static_assert(!is_const<T>::value,
                  "A const DMK tensor cannot become a mutable Torch tensor");
    TORCH_CHECK(out.defined(),
                "to_torch requires an existing Torch tensor owning the storage");
    TORCH_CHECK(out.is_cuda(), "to_torch requires a DCU/CUDA owner");
    TORCH_CHECK(out.layout() == at::kStrided,
                "to_torch requires a strided owner");
    TORCH_CHECK(out.scalar_type() == c10::CppTypeToScalarType<T>::value,
                "to_torch dtype mismatch");
    TORCH_CHECK(out.data_ptr<T>() == src.data,
                "to_torch owner must have the same starting data pointer");

    size_t count = 0;
    TORCH_CHECK(src.shape.try_numel(count), "to_torch shape overflow");
    TORCH_CHECK(count == 0 || src.data != nullptr,
                "to_torch requires a non-null pointer for nonempty data");
    vector<int64_t> sizes(Rank);
    vector<int64_t> strides(Rank);
    constexpr size_t max_torch_index =
        static_cast<size_t>(numeric_limits<int64_t>::max());
    for (size_t axis = 0; axis < Rank; ++axis) {
        TORCH_CHECK(src.shape.dims[axis] <= max_torch_index &&
                    src.layout.strides[axis] <= max_torch_index,
                    "to_torch shape or stride exceeds int64 range");
        sizes[axis] = static_cast<int64_t>(src.shape.dims[axis]);
        strides[axis] = static_cast<int64_t>(src.layout.strides[axis]);
    }
    out = out.as_strided(sizes, strides);
}
#endif  // __HIP_DEVICE_COMPILE__

}  // namespace dmk::tensor
