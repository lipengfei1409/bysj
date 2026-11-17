#pragma once
#include "util.h"

namespace dmk {

// Construct derived objects on the device before calling through BaseOp*.
// A host-constructed polymorphic object cannot be copied to the device and used there.
class BaseOp {
public:
    __device__ virtual ~BaseOp() = default;

    __device__ virtual void shard() = 0;
    __device__ virtual void loader() = 0;
    __device__ virtual void consumer() = 0;
    __device__ virtual void storer() = 0;
private:

};

}  // namespace dmk
