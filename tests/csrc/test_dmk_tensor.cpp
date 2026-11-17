// Standalone metadata test. Build with:
// hipcc -std=c++17 -I csrc/dmk tests/csrc/test_dmk_tensor.cpp -o /tmp/test_dmk_tensor
#include "tensor.h"

namespace {

void test_shape_and_strides() {
    using namespace dmk;
    const Shape<3> shape{{2, 3, 4}};
    assert(shape.ndim() == 3);
    assert(shape.numel() == 24);

    const auto layout = HbmLayout<3>::contiguous(shape);
    assert(layout.strides[0] == 12);
    assert(layout.strides[1] == 4);
    assert(layout.strides[2] == 1);
    assert(layout.offset(shape, Shape<3>{{1, 2, 3}}) == 23);

    size_t count = 0;
    const Shape<2> too_many{{~size_t{0}, 2}};
    const Shape<2> empty{{0, ~size_t{0}}};
    assert(!too_many.try_numel(count));
    assert(empty.try_numel(count) && count == 0);
}

void test_view_operations() {
    using namespace dmk;
    const Shape<2> shape{{2, 3}};
    Tensor<float, 2, MemorySpace::HBM> matrix{
        nullptr, shape, HbmLayout<2>::contiguous(shape)};

    assert(matrix.is_contiguous());
    assert(matrix.offset(Shape<2>{{1, 2}}) == 5);

    auto transposed = matrix.transpose(0, 1);
    assert(transposed.data == matrix.data);
    assert(transposed.shape.dims[0] == 3 && transposed.shape.dims[1] == 2);
    assert(transposed.offset(Shape<2>{{2, 1}}) == 5);
    assert(!transposed.is_contiguous());

    Tensor<float, 1, MemorySpace::HBM> flat{};
    assert(matrix.reshape(Shape<1>{{6}}, flat));
    assert(flat.data == matrix.data && flat.shape.dims[0] == 6);
    assert(flat.offset(Shape<1>{{5}}) == 5);
    assert(!transposed.reshape(Shape<1>{{6}}, flat));
    assert(!matrix.reshape(Shape<1>{{5}}, flat));
}

template <typename T>
void test_swizzle() {
    using namespace dmk;
    const Shape<2> tile{{32, 37}};
    const ShmemSwizzleLayout<sizeof(T)> layout{};
    const size_t capacity = layout.required_elements(tile);
    size_t checked_capacity = 0;
    assert(layout.try_required_elements(tile, checked_capacity));
    assert(checked_capacity == capacity);

    Tensor<T, 2, MemorySpace::Shmem, ShmemSwizzleLayout<sizeof(T)>> view{
        nullptr, tile, layout};
    vector<bool> used(capacity, false);
    bool banks[32]{};
    for (size_t row = 0; row < tile.dims[0]; ++row) {
        for (size_t col = 0; col < tile.dims[1]; ++col) {
            const size_t index = view.offset(Shape<2>{{row, col}});
            assert(index < capacity && !used[index]);
            used[index] = true;
        }
        const size_t first = view.offset(Shape<2>{{row, 0}});
        const size_t bank = (first * sizeof(T) / 4) % 32;
        assert(!banks[bank]);
        banks[bank] = true;
    }
    const Shape<2> too_large{{~size_t{0}, 1}};
    assert(!layout.try_required_elements(too_large, checked_capacity));
}

struct TestFragment { float value; };

void test_register_fragment() {
    using namespace dmk;
    Tensor<TestFragment, 2, MemorySpace::Register> fragment{
        {1.0f}, Shape<2>{{16, 16}}, {}};
    assert(fragment.numel() == 256);
    assert(fragment.fragment.value == 1.0f);
}

}  // namespace

int main() {
    test_shape_and_strides();
    test_view_operations();
    test_swizzle<uint8_t>();
    test_swizzle<uint16_t>();
    test_swizzle<uint32_t>();
    test_register_fragment();
}
