// Standalone layout-copy test; no GPU allocation is needed.
// Build: hipcc -std=c++17 -I csrc/dmk tests/csrc/test_dmk_tensor_copy.cpp -o /tmp/test_dmk_tensor_copy
#include "copy.h"

int main() {
    using namespace dmk;
    const Shape<2> shape{{2, 3}};
    const float input[6] = {1, 2, 3, 4, 5, 6};
    float column_storage[6]{};
    Tensor<const float, 2, MemorySpace::HBM> src{
        input, shape, HbmLayout<2>::contiguous(shape)};
    Tensor<float, 2, MemorySpace::HBM> column{
        column_storage, shape, StridedLayout<2>{{1, 2}}};

    for (size_t worker = 0; worker < 3; ++worker)
        assert(copy::tensor_copy(column, src, worker, 3));
    const float column_expected[6] = {1, 4, 2, 5, 3, 6};
    for (size_t i = 0; i < 6; ++i) assert(column_storage[i] == column_expected[i]);

    float roundtrip[6]{};
    Tensor<float, 2, MemorySpace::HBM> restored{
        roundtrip, shape, HbmLayout<2>::contiguous(shape)};
    for (size_t worker = 0; worker < 4; ++worker)
        assert(copy::tensor_copy(restored, column, worker, 4));
    for (size_t i = 0; i < 6; ++i) assert(roundtrip[i] == input[i]);

    const Shape<2> tile_shape{{32, 37}};
    vector<float> tile_input(tile_shape.numel());
    for (size_t i = 0; i < tile_input.size(); ++i)
        tile_input[i] = static_cast<float>(i + 1);
    const ShmemSwizzleLayout<4> swizzle{};
    vector<float> physical(swizzle.required_elements(tile_shape), -1.0f);
    vector<float> tile_output(tile_shape.numel(), 0.0f);
    Tensor<const float, 2, MemorySpace::HBM> tile_src{
        tile_input.data(), tile_shape, HbmLayout<2>::contiguous(tile_shape)};
    Tensor<float, 2, MemorySpace::Shmem, ShmemSwizzleLayout<4>> tile_dst{
        physical.data(), tile_shape, swizzle};
    Tensor<float, 2, MemorySpace::HBM> tile_result{
        tile_output.data(), tile_shape, HbmLayout<2>::contiguous(tile_shape)};
    for (size_t worker = 0; worker < 64; ++worker)
        assert(copy::global_to_lds(tile_dst, tile_src, worker, 64));
    for (size_t worker = 0; worker < 64; ++worker)
        assert(copy::lds_to_global(tile_result, tile_dst, worker, 64));
    for (size_t i = 0; i < tile_input.size(); ++i)
        assert(tile_output[i] == tile_input[i]);

    float sentinel[4] = {-1, -1, -1, -1};
    const Shape<2> wrong_shape{{2, 2}};
    Tensor<float, 2, MemorySpace::HBM> wrong{
        sentinel, wrong_shape, HbmLayout<2>::contiguous(wrong_shape)};
    assert(!copy::tensor_copy(wrong, src, 0, 1));
    for (float value : sentinel) assert(value == -1);
    assert(!copy::tensor_copy(restored, src, 0, 0));
    assert(!copy::tensor_copy(restored, src, 1, 1));
    Tensor<float, 2, MemorySpace::HBM> missing{
        nullptr, shape, HbmLayout<2>::contiguous(shape)};
    assert(!copy::tensor_copy(missing, src, 0, 1));

    const Shape<3> volume_shape{{2, 3, 4}};
    float volume_input[24]{};
    float volume_output[24]{};
    for (size_t i = 0; i < 24; ++i) volume_input[i] = static_cast<float>(i);
    Tensor<const float, 3, MemorySpace::HBM> volume_src{
        volume_input, volume_shape, HbmLayout<3>::contiguous(volume_shape)};
    Tensor<float, 3, MemorySpace::HBM> volume_dst{
        volume_output, volume_shape, HbmLayout<3>::contiguous(volume_shape)};
    for (size_t worker = 0; worker < 5; ++worker)
        assert(copy::tensor_copy(volume_dst, volume_src, worker, 5));
    for (size_t i = 0; i < 24; ++i)
        assert(volume_output[i] == volume_input[i]);

    // With a column-major global view, neighboring workers must touch
    // neighboring global addresses rather than neighboring logical columns.
    const Shape<2> column_shape{{3, 2}};
    const StridedLayout<2> column_layout{{1, 3}};
    const float column_major_values[6] = {1, 2, 3, 4, 5, 6};
    float lds_values[6] = {-1, -1, -1, -1, -1, -1};
    Tensor<const float, 2, MemorySpace::HBM> column_global{
        column_major_values, column_shape, column_layout};
    Tensor<float, 2, MemorySpace::Shmem> lds_plain{
        lds_values, column_shape, ShmemStridedLayout<2>::contiguous(column_shape)};
    assert(copy::global_to_lds(lds_plain, column_global, 0, 6));
    assert(copy::global_to_lds(lds_plain, column_global, 1, 6));
    assert(lds_values[0] == 1 && lds_values[2] == 2 && lds_values[1] == -1);

    const float lds_source_values[6] = {10, 11, 12, 13, 14, 15};
    float global_values[6] = {-1, -1, -1, -1, -1, -1};
    Tensor<const float, 2, MemorySpace::Shmem> lds_source{
        lds_source_values, column_shape,
        ShmemStridedLayout<2>::contiguous(column_shape)};
    Tensor<float, 2, MemorySpace::HBM> column_global_dst{
        global_values, column_shape, column_layout};
    assert(copy::lds_to_global(column_global_dst, lds_source, 0, 6));
    assert(copy::lds_to_global(column_global_dst, lds_source, 1, 6));
    assert(global_values[0] == 10 && global_values[1] == 12 && global_values[3] == -1);

    const Shape<2> empty_shape{{0, 3}};
    Tensor<const float, 2, MemorySpace::HBM> empty_src{
        nullptr, empty_shape, HbmLayout<2>::contiguous(empty_shape)};
    Tensor<float, 2, MemorySpace::Shmem> empty_dst{
        nullptr, empty_shape, ShmemStridedLayout<2>::contiguous(empty_shape)};
    assert(copy::tensor_copy(empty_dst, empty_src, 0, 1));
}
