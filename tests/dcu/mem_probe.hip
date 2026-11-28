// Memory-path bandwidth probe for DCU: HBM -> LDS and LDS -> registers.
//
// Three measurements, each with its own bandwidth number:
//
//   hbm_read      global float4 loads, grid-stride, 4 in flight   -> HBM read GB/s
//   hbm_to_lds    global float4 -> shared float4, 16 KiB chunks   -> HBM read GB/s
//                 (the LDS write side is ~8x faster than HBM per CU,
//                  so this number should track hbm_read)
//   lds_read      shared float4 -> registers, rotating address      -> B/cycle/CU
//                 (conflict-free: lane i reads 16 B at i*16 B, so
//                  a wave64 covers 1 KiB contiguous = 32 B/bank)
//   lds_conflict  same but lane i reads at i*256 B (all lanes hit
//                 the same 4 banks)                                 -> shows the 32-bank limit
//
// Build:  hipcc -O3 -std=c++17 tests/dcu/mem_probe.hip -o /tmp/mem_probe
// Run:    /tmp/mem_probe --mb 512 --blocks 480 --iters 200000
//
// Notes:
//   * HBM tests want several blocks per CU (--blocks 480 = 4/CU) to keep enough
//     loads in flight; LDS tests want exactly one block per CU (--blocks 120).
//   * LDS numbers are reported as bytes/cycle/CU, measured with clock64 inside
//     the kernel, plus GB/s from host events. Theoretical ceiling is 128 B/cycle
//     (32 banks x 4 B); 16 B/lane x 64 lanes = 1 KiB per wave instruction, so a
//     perfect wave takes 8 cycles.
//   * The conflict variant should land near 1/8 of the conflict-free rate.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t status = (call);                                               \
    if (status != hipSuccess) {                                               \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                   hipGetErrorString(status));                                \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

constexpr int kWave = 64;
constexpr int kLdsFloat4 = 1024;  // 16 KiB of float4

__global__ void hbm_read_kernel(const float4* __restrict__ src, size_t n4,
                                float* __restrict__ sink) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

  for (; i + 3 * stride < n4; i += 4 * stride) {
    const float4 v0 = src[i];
    const float4 v1 = src[i + stride];
    const float4 v2 = src[i + 2 * stride];
    const float4 v3 = src[i + 3 * stride];
    acc.x += v0.x + v1.x + v2.x + v3.x;
    acc.y += v0.y + v1.y + v2.y + v3.y;
    acc.z += v0.z + v1.z + v2.z + v3.z;
    acc.w += v0.w + v1.w + v2.w + v3.w;
  }
  for (; i < n4; i += stride) {
    const float4 v = src[i];
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }
  if (acc.x + acc.y + acc.z + acc.w == 12345.678f) sink[0] = acc.x;
}

__global__ void hbm_to_lds_kernel(const float4* __restrict__ src, size_t n4,
                                  float* __restrict__ sink) {
  extern __shared__ float4 smem[];
  float acc = 0.f;

  const size_t chunks = n4 / kLdsFloat4;
  for (size_t chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
    const float4* base = src + chunk * kLdsFloat4;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int k = u * blockDim.x + threadIdx.x;
      if (k < kLdsFloat4) smem[k] = base[k];
    }
    __syncthreads();
    acc += smem[threadIdx.x].x;  // touch the page so the stores are not dead
    __syncthreads();
  }
  if (acc == 12345.678f) sink[0] = acc;
}

// Shared loads: wave w owns a 1 KiB slice, lane l takes float4 (w*64 + l),
// advancing 64 float4 per iteration.  All addresses are distinct and the
// sequence is 1 KiB contiguous per wave -> no bank conflicts.
__global__ void lds_read_kernel(int iters, int stride4, float* __restrict__ sink,
                                unsigned long long* __restrict__ cycles) {
  __shared__ float4 smem[kLdsFloat4];
  for (int i = threadIdx.x; i < kLdsFloat4; i += blockDim.x)
    smem[i] = make_float4(static_cast<float>(i), 1.f, 2.f, 3.f);
  __syncthreads();

  (void)stride4;
  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  constexpr int kMask = kLdsFloat4 - 1;

  const unsigned long long t0 = clock64();
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    const int idx = (wave * kWave + i * kWave + lane) & kMask;
    const float4 v = smem[idx];
    // Keep the 16 B load without spending VALU instructions on it: the empty
    // asm with VGPR constraints forces all four words into registers, so the
    // loop measures LDS bandwidth instead of FADD issue.
    asm volatile("" : : "v"(v.x), "v"(v.y), "v"(v.z), "v"(v.w));
  }
  const unsigned long long t1 = clock64();

  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
  if (smem[0].x == 12345.678f) sink[0] = 1.f;
}

// Same loop, but lane l reads at l*16 float4 (256 B apart): all 64 lanes hit
// the four banks {0,1,2,3}, so the 32-bank interleave degrades to a serial
// replay.  Use it as the "bad pattern" reference.
__global__ void lds_conflict_kernel(int iters, float* __restrict__ sink,
                                    unsigned long long* __restrict__ cycles) {
  __shared__ float4 smem[kLdsFloat4];
  for (int i = threadIdx.x; i < kLdsFloat4; i += blockDim.x)
    smem[i] = make_float4(static_cast<float>(i), 1.f, 2.f, 3.f);
  __syncthreads();

  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  constexpr int kMask = kLdsFloat4 - 1;

  const unsigned long long t0 = clock64();
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    const int idx = (wave * kWave + i * kWave + lane * 16) & kMask;
    const float4 v = smem[idx];
    asm volatile("" : : "v"(v.x), "v"(v.y), "v"(v.z), "v"(v.w));
  }
  const unsigned long long t1 = clock64();

  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
  if (smem[0].x == 12345.678f) sink[0] = 1.f;
}

struct Args {
  int mb = 512;
  int blocks = 480;
  int threads = 256;
  int iters = 200000;
  int reps = 20;
  int lds_blocks = 120;
};

double time_ms(void (*launch)(void*), void* ctx, int reps) {
  hipEvent_t begin, end;
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    HIP_CHECK(hipEventRecord(begin));
    launch(ctx);
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
    best = std::min(best, static_cast<double>(ms));
  }
  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
  return best;
}

struct HbmCtx {
  const float4* src;
  size_t n4;
  float* sink;
  int blocks;
  int threads;
  int which;  // 0 = read, 1 = to_lds
};

void launch_hbm(void* raw) {
  HbmCtx* c = static_cast<HbmCtx*>(raw);
  if (c->which == 0) {
    hbm_read_kernel<<<c->blocks, c->threads>>>(c->src, c->n4, c->sink);
  } else {
    // The copy kernel assumes 256 threads (4 float4 per thread = 16 KiB).
    hbm_to_lds_kernel<<<c->blocks, 256, 16 * 1024>>>(c->src, c->n4, c->sink);
  }
}

struct LdsCtx {
  int iters;
  int blocks;
  int threads;
  int stride4;
  float* sink;
  unsigned long long* cycles;
  int conflict;
};

void launch_lds(void* raw) {
  LdsCtx* c = static_cast<LdsCtx*>(raw);
  if (c->conflict) {
    lds_conflict_kernel<<<c->blocks, c->threads>>>(c->iters, c->sink, c->cycles);
  } else {
    lds_read_kernel<<<c->blocks, c->threads>>>(c->iters, c->stride4, c->sink,
                                               c->cycles);
  }
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const char* k = argv[i];
    auto next = [&]() -> int {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", k);
        std::exit(2);
      }
      return std::atoi(argv[++i]);
    };
    if (!std::strcmp(k, "--mb")) a.mb = next();
    else if (!std::strcmp(k, "--blocks")) a.blocks = next();
    else if (!std::strcmp(k, "--threads")) a.threads = next();
    else if (!std::strcmp(k, "--iters")) a.iters = next();
    else if (!std::strcmp(k, "--reps")) a.reps = next();
    else if (!std::strcmp(k, "--lds-blocks")) a.lds_blocks = next();
    else {
      std::fprintf(stderr, "unknown option %s\n", k);
      std::exit(2);
    }
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);

  int device = 0;
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  const int cus = prop.multiProcessorCount;
  std::printf("device: %s  CUs=%d  clock=%.2f GHz  LDS/CU=%.0f KiB\n",
              prop.name, cus, prop.clockRate * 1e-6,
              static_cast<double>(prop.maxSharedMemoryPerMultiProcessor) / 1024.0);

  const size_t bytes = static_cast<size_t>(args.mb) * 1024 * 1024;
  const size_t n4 = bytes / sizeof(float4);
  std::vector<float> host(4, 0.f);

  float* src = nullptr;
  float* sink = nullptr;
  unsigned long long* cycles = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&src), bytes));
  HIP_CHECK(hipMemset(src, 0x3f, bytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sink), sizeof(float)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&cycles),
                      sizeof(unsigned long long) * args.lds_blocks));

  const float4* src4 = reinterpret_cast<const float4*>(src);

  // ---- HBM read ------------------------------------------------------------
  HbmCtx read_ctx{src4, n4, sink, args.blocks, args.threads, 0};
  const double read_ms = time_ms(launch_hbm, &read_ctx, args.reps);
  const double read_gbs = bytes / (read_ms * 1e-3) / 1e9;

  // ---- HBM -> LDS ----------------------------------------------------------
  HbmCtx copy_ctx{src4, n4, sink, args.blocks, args.threads, 1};
  const double copy_ms = time_ms(launch_hbm, &copy_ctx, args.reps);
  const double copy_gbs = bytes / (copy_ms * 1e-3) / 1e9;

  // ---- LDS -> registers (conflict-free) ------------------------------------
  LdsCtx lds_ctx{args.iters, args.lds_blocks, args.threads, kWave, sink, cycles,
                 0};
  HIP_CHECK(hipMemset(sink, 0, sizeof(float)));
  const double lds_ms = time_ms(launch_lds, &lds_ctx, args.reps);

  std::vector<unsigned long long> host_cycles(args.lds_blocks, 0);
  HIP_CHECK(hipMemcpy(host_cycles.data(), cycles,
                      sizeof(unsigned long long) * args.lds_blocks,
                      hipMemcpyDeviceToHost));
  unsigned long long cycle_sum = 0;
  for (auto c : host_cycles) cycle_sum += c;
  const double cycle_mean =
      static_cast<double>(cycle_sum) / std::max(1, args.lds_blocks);

  const double lds_bytes_per_block =
      static_cast<double>(args.iters) * args.threads * sizeof(float4);
  const double lds_bytes = lds_bytes_per_block * args.lds_blocks;
  const double lds_gbs = lds_bytes / (lds_ms * 1e-3) / 1e9;
  const double lds_bpc = lds_bytes_per_block / cycle_mean;

  // ---- LDS -> registers (conflicting) --------------------------------------
  LdsCtx conflict_ctx{args.iters, args.lds_blocks, args.threads, kWave, sink,
                      cycles, 1};
  const double conflict_ms = time_ms(launch_lds, &conflict_ctx, args.reps);
  HIP_CHECK(hipMemcpy(host_cycles.data(), cycles,
                      sizeof(unsigned long long) * args.lds_blocks,
                      hipMemcpyDeviceToHost));
  cycle_sum = 0;
  for (auto c : host_cycles) cycle_sum += c;
  const double conflict_cycle_mean =
      static_cast<double>(cycle_sum) / std::max(1, args.lds_blocks);
  const double conflict_gbs = lds_bytes / (conflict_ms * 1e-3) / 1e9;
  const double conflict_bpc = lds_bytes_per_block / conflict_cycle_mean;

  std::printf("\n%-22s %12s %14s %12s\n", "test", "GB/s", "bytes/cycle/CU",
              "note");
  std::printf("%-22s %12.1f %14s %12s\n", "HBM read", read_gbs, "-",
              "grid-stride");
  std::printf("%-22s %12.1f %14s %12s\n", "HBM -> LDS", copy_gbs, "-",
              "16 KiB chunks");
  std::printf("%-22s %12.1f %14.2f %12s\n", "LDS -> reg", lds_gbs, lds_bpc,
              "16 B/lane, 8 cyc");
  std::printf("%-22s %12.1f %14.2f %12s\n", "LDS -> reg (conflict)",
              conflict_gbs, conflict_bpc, "all lanes same bank");
  std::printf("\nHBM per CU: %.1f GB/s   LDS aggregate: %.1f TB/s\n",
              read_gbs / cus, lds_gbs / 1000.0);
  std::printf("LDS ceiling 128 B/cycle/CU = %.1f GB/s/CU at %.2f GHz\n",
              128.0 * prop.clockRate * 1e3 / 1e9, prop.clockRate * 1e-6);

  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(sink));
  HIP_CHECK(hipFree(cycles));
  (void)host;
  return 0;
}
