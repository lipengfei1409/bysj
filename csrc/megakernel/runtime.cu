// Persistent HIP workers for explicitly selected DPA4 edge-feature tasks.
#include "../include/ops.h"
#include "../include/checks.h"
#include "../include/megakernel/edge_features.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

namespace {

constexpr int64_t kMetaWidth = 8;
constexpr int64_t kTicketWidth = 4;
constexpr int kEdgeFeaturesOpcode = 1;
constexpr int kEdgeGeometryOpcode = 2;
constexpr int kRadialBasisOpcode = 3;
// Each tile is enqueued once; capacity is the total tile count, so slots are
// sequence-tagged append-only for the lifetime of one invocation.
enum RuntimeError : int {
  kOk = 0,
  kStaleGeneration = 1,
  kUnsupportedOpcode = 2,
  kQueueOverflow = 3,
  kInvalidTask = 4,
  kStalled = 5,
};

__device__ __forceinline__ unsigned long long atomic_load_u64(const int64_t* p) {
  auto* ptr = reinterpret_cast<unsigned long long*>(const_cast<int64_t*>(p));
  return atomicCAS(ptr, 0ULL, 0ULL);
}

__device__ __forceinline__ void set_error(int32_t* status, int code) {
  atomicCAS(status, kOk, code);
}

__device__ __forceinline__ int load_status(const int32_t* status) {
  return static_cast<int>(atomicAdd(
      reinterpret_cast<unsigned int*>(const_cast<int32_t*>(status)), 0U));
}

__device__ __forceinline__ bool enqueue(
    int64_t* queue, int64_t capacity, int64_t* control, int32_t* status,
    int64_t task, int64_t tile, int64_t generation) {
  const unsigned long long pos = atomicAdd(
      reinterpret_cast<unsigned long long*>(control), 1ULL);
  if (pos >= static_cast<unsigned long long>(capacity)) {
    set_error(status, kQueueOverflow);
    return false;
  }
  int64_t* slot = queue + pos * kTicketWidth;
  slot[1] = task;
  slot[2] = tile;
  slot[3] = generation;
  __threadfence();
  atomicCAS(reinterpret_cast<unsigned long long*>(slot), 0ULL, pos + 1ULL);
  return true;
}

__device__ __forceinline__ bool dequeue(
    int64_t* queue, int64_t* control, int64_t capacity, int64_t generation,
    int32_t* status, int64_t& task, int64_t& tile) {
  const unsigned long long pos = atomic_load_u64(control + 1);
  const unsigned long long published = atomic_load_u64(control);
  if (pos >= published || pos >= static_cast<unsigned long long>(capacity)) return false;
  int64_t* slot = queue + pos * kTicketWidth;
  if (atomic_load_u64(slot) != pos + 1ULL) return false;
  auto* dequeue_pos = reinterpret_cast<unsigned long long*>(control + 1);
  if (atomicCAS(dequeue_pos, pos, pos + 1ULL) != pos) return false;
  task = slot[1];
  tile = slot[2];
  if (static_cast<unsigned long long>(slot[3]) !=
      static_cast<unsigned long long>(generation)) {
    set_error(status, kStaleGeneration);
    return false;
  }
  return true;
}

__global__ void initialize_runtime_kernel(
    const int64_t* meta, int64_t n_tasks, const int64_t* roots,
    int64_t n_roots, int64_t total_tickets, int64_t generation,
    int64_t* queue, int64_t queue_capacity, int64_t* states,
    int64_t* control, int32_t* status) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid == 0) {
    control[2] = total_tickets;
    control[3] = 0;
  }
  if (tid < n_tasks) {
    states[3 * tid] = meta[tid * kMetaWidth + 4];
    states[3 * tid + 1] = meta[tid * kMetaWidth + 3];
    states[3 * tid + 2] = generation;
  }
  if (tid < n_roots) {
    enqueue(queue, queue_capacity, control, status,
            roots[2 * tid], roots[2 * tid + 1], generation);
  }
}

__global__ void persistent_edge_features_worker_kernel(
    dcu::megakernel::EdgeFeaturesArgs args, const int64_t* meta,
    int64_t n_tasks, const int64_t* dependent_offsets,
    const int64_t* dependent_ids, int64_t* queue, int64_t queue_capacity,
    int64_t* states, int64_t* control, int32_t* status, int64_t generation) {
  __shared__ int has_ticket;
  __shared__ int stop_worker;
  __shared__ int64_t task_id;
  __shared__ int64_t tile_id;

  while (true) {
    if (threadIdx.x == 0) {
      has_ticket = 0;
      stop_worker = load_status(status) != kOk;
      if (!stop_worker) {
        atomicAdd(reinterpret_cast<unsigned long long*>(control + 3), 1ULL);
        has_ticket = dequeue(queue, control, queue_capacity, generation,
                             status, task_id, tile_id) ? 1 : 0;
        if (!has_ticket) {
          const unsigned long long active_before = atomicAdd(
              reinterpret_cast<unsigned long long*>(control + 3), ~0ULL);
          const unsigned long long remaining = atomic_load_u64(control + 2);
          if (remaining == 0 || load_status(status) != kOk) {
            stop_worker = 1;
          } else if (active_before == 1 &&
                     atomic_load_u64(control + 1) >= atomic_load_u64(control)) {
            set_error(status, kStalled);
            stop_worker = 1;
          }
        }
      }
    }
    __syncthreads();
    if (stop_worker) break;
    if (!has_ticket) continue;

    if (task_id < 0 || task_id >= n_tasks) {
      if (threadIdx.x == 0) {
        set_error(status, kInvalidTask);
        atomicAdd(reinterpret_cast<unsigned long long*>(control + 3), ~0ULL);
      }
      __syncthreads();
      break;
    }
    const int64_t* task = meta + task_id * kMetaWidth;
    const int64_t opcode = task[0];
    const int64_t work_items = task[1];
    const int64_t tile_size = task[2];
    if (tile_id < 0 || tile_id >= task[3]) {
      if (threadIdx.x == 0) {
        set_error(status, kInvalidTask);
        atomicAdd(reinterpret_cast<unsigned long long*>(control + 3), ~0ULL);
      }
      __syncthreads();
      break;
    }
    const int64_t begin = tile_id * tile_size;
    const int64_t remaining_in_task = work_items - begin;
    const int64_t tile_work = tile_size < remaining_in_task ? tile_size : remaining_in_task;
    const int64_t end = begin + tile_work;
    if (opcode != kEdgeFeaturesOpcode && opcode != kEdgeGeometryOpcode
        && opcode != kRadialBasisOpcode) {
      if (threadIdx.x == 0) {
        set_error(status, kUnsupportedOpcode);
        atomicAdd(reinterpret_cast<unsigned long long*>(control + 3), ~0ULL);
      }
      __syncthreads();
      break;
    }
    for (int64_t edge = begin + threadIdx.x; edge < end; edge += blockDim.x) {
      if (opcode == kEdgeFeaturesOpcode) {
        dcu::megakernel::edge_features_one(args, edge);
      } else if (opcode == kEdgeGeometryOpcode) {
        dcu::megakernel::edge_geometry_one(args, edge);
      } else {
        dcu::megakernel::radial_features_one(args, edge);
      }
    }
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
      if (states[3 * task_id + 2] != generation) {
        set_error(status, kStaleGeneration);
      } else {
        const unsigned long long tiles_before = atomicAdd(
            reinterpret_cast<unsigned long long*>(states + 3 * task_id + 1), ~0ULL);
        if (tiles_before == 1ULL) {
          const int64_t dependent_begin = task[5];
          const int64_t dependent_end = dependent_offsets[task_id + 1];
          if (dependent_offsets[task_id] != dependent_begin || dependent_end < dependent_begin) {
            set_error(status, kInvalidTask);
          } else {
            for (int64_t i = dependent_begin; i < dependent_end; ++i) {
              const int64_t child = dependent_ids[i];
              if (child < 0 || child >= n_tasks || states[3 * child + 2] != generation) {
                set_error(status, kStaleGeneration);
                continue;
              }
              const unsigned long long parents_before = atomicAdd(
                  reinterpret_cast<unsigned long long*>(states + 3 * child), ~0ULL);
              if (parents_before == 1ULL) {
                const int64_t child_tiles = meta[child * kMetaWidth + 3];
                for (int64_t tile = 0; tile < child_tiles; ++tile) {
                  enqueue(queue, queue_capacity, control, status, child, tile, generation);
                }
              }
            }
          }
        }
      }
      atomicAdd(reinterpret_cast<unsigned long long*>(control + 2), ~0ULL);
      atomicAdd(reinterpret_cast<unsigned long long*>(control + 3), ~0ULL);
    }
    __syncthreads();
  }
}


void validate_runtime_plan(
    const std::vector<int64_t>& opcodes, const std::vector<int64_t>& epochs,
    const std::vector<int64_t>& work_items, const std::vector<int64_t>& tile_sizes,
    const std::vector<int64_t>& dependency_offsets,
    const std::vector<int64_t>& dependency_ids, int64_t edge_count,
    int64_t plan_version, int64_t generation, int64_t queue_capacity,
    int64_t max_queue_tiles) {
  const size_t n = opcodes.size();
  TORCH_CHECK(plan_version == 1, "unsupported persistent runtime-plan version");
  TORCH_CHECK(n > 0 && epochs.size() == n && work_items.size() == n && tile_sizes.size() == n,
              "runtime task arrays must be nonempty and have matching lengths");
  TORCH_CHECK(dependency_offsets.size() == n + 1 && dependency_offsets.front() == 0
              && dependency_offsets.back() == static_cast<int64_t>(dependency_ids.size()),
              "runtime dependency offsets are invalid");
  TORCH_CHECK(generation > 0 && generation < (1LL << 31), "runtime generation is out of range");
  TORCH_CHECK(max_queue_tiles > 0, "max_queue_tiles must be positive");
  std::vector<int64_t> indegree(n, 0);
  std::vector<std::vector<int64_t>> children(n);
  int64_t expected_tiles = 0;
  for (size_t i = 0; i < n; ++i) {
    TORCH_CHECK(opcodes[i] == kEdgeFeaturesOpcode || opcodes[i] == kEdgeGeometryOpcode
                || opcodes[i] == kRadialBasisOpcode,
                "unsupported persistent HIP opcode ", opcodes[i], " at task ", i);
    TORCH_CHECK(epochs[i] >= 0 && work_items[i] == edge_count && tile_sizes[i] > 0,
                "runtime edge-feature task dimensions disagree with the payload");
    TORCH_CHECK(dependency_offsets[i] >= 0
                && dependency_offsets[i] <= dependency_offsets[i + 1],
                "runtime dependency offsets must be monotone");
    const int64_t tiles = edge_count / tile_sizes[i] + (edge_count % tile_sizes[i] != 0);
    TORCH_CHECK(tiles <= std::numeric_limits<int64_t>::max() - expected_tiles,
                "runtime tile count overflow");
    expected_tiles += tiles;
    indegree[i] = dependency_offsets[i + 1] - dependency_offsets[i];
    std::unordered_set<int64_t> seen_dependencies;
    for (int64_t j = dependency_offsets[i]; j < dependency_offsets[i + 1]; ++j) {
      const int64_t parent = dependency_ids[j];
      TORCH_CHECK(parent >= 0 && parent < static_cast<int64_t>(n) && parent != static_cast<int64_t>(i),
                  "runtime dependency id is invalid");
      TORCH_CHECK(seen_dependencies.insert(parent).second,
                  "runtime task repeats a dependency");
      children[parent].push_back(static_cast<int64_t>(i));
    }
  }
  TORCH_CHECK(queue_capacity == expected_tiles,
              "runtime queue capacity must equal the total tile ticket count");
  TORCH_CHECK(queue_capacity <= max_queue_tiles,
              "runtime queue capacity exceeds the configured limit");
  std::vector<int64_t> ready;
  for (size_t i = 0; i < n; ++i) if (indegree[i] == 0) ready.push_back(i);
  size_t visited = 0;
  while (!ready.empty()) {
    const int64_t task = ready.back();
    ready.pop_back();
    ++visited;
    for (const int64_t child : children[task]) if (--indegree[child] == 0) ready.push_back(child);
  }
  TORCH_CHECK(visited == n, "runtime task graph contains a dependency cycle");
  if (n == 1) {
    TORCH_CHECK(opcodes[0] == kEdgeFeaturesOpcode && dependency_offsets[1] == 0,
                "a single-task edge-feature plan must use the fused edge_features opcode");
  } else {
    TORCH_CHECK(n == 2 && opcodes[0] == kEdgeGeometryOpcode
                && opcodes[1] == kRadialBasisOpcode
                && dependency_offsets[1] == 0
                && dependency_offsets[2] - dependency_offsets[1] == 1
                && dependency_ids[dependency_offsets[1]] == 0,
                "runtime supports only fused edge_features or edge_geometry -> radial_basis");
  }
}

}  // namespace

std::vector<Tensor> edge_features_persistent_hip(
    Tensor positions, Tensor edge_index, Tensor shifts, Tensor cells,
    Tensor edge_frame, Tensor mask, Tensor frequencies, double cutoff,
    int64_t basis_exponent, int64_t edge_exponent, double eps,
    std::vector<int64_t> opcodes, std::vector<int64_t> epochs,
    std::vector<int64_t> work_items, std::vector<int64_t> tile_sizes,
    std::vector<int64_t> dependency_offsets, std::vector<int64_t> dependency_ids,
    int64_t plan_version, int64_t generation, int64_t queue_capacity,
    int64_t max_queue_tiles) {
  f32(positions); f32(cells); f32(frequencies);
  i64(edge_index); i64(shifts); i64(edge_frame);
  TORCH_CHECK(mask.scalar_type() == torch::kBool && mask.is_cuda() && mask.is_contiguous(),
              "edge mask must be a contiguous DCU bool tensor");
  same_device(positions, {edge_index, shifts, cells, edge_frame, mask, frequencies});
  TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must be [N,3]");
  TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2, "edge_index must be [2,E]");
  const int64_t n_edges = edge_index.size(1), n_radial = frequencies.size(1);
  TORCH_CHECK(shifts.dim() == 2 && shifts.size(0) == n_edges && shifts.size(1) == 3
              && edge_frame.dim() == 1 && edge_frame.numel() == n_edges
              && mask.dim() == 1 && mask.numel() == n_edges,
              "edge topology shapes disagree");
  TORCH_CHECK(cells.dim() == 3 && cells.size(0) > 0 && cells.size(1) == 3 && cells.size(2) == 3,
              "cells must be [F,3,3]");
  TORCH_CHECK(frequencies.dim() == 2 && frequencies.size(0) == 1 && n_radial > 0,
              "frequencies must be [1,R]");
  TORCH_CHECK(cutoff > 0.0 && std::isfinite(cutoff) && eps > 0.0 && std::isfinite(eps)
              && basis_exponent >= 0 && basis_exponent <= 64
              && edge_exponent >= 0 && edge_exponent <= 64,
              "invalid edge-feature parameters");
  validate_runtime_plan(opcodes, epochs, work_items, tile_sizes,
                        dependency_offsets, dependency_ids, n_edges,
                        plan_version, generation, queue_capacity, max_queue_tiles);

  c10::cuda::CUDAGuard guard(positions.device());
  auto vec = torch::empty({n_edges, 3}, positions.options());
  auto distance = torch::empty({n_edges, 1}, positions.options());
  auto unit = torch::empty({n_edges, 3}, positions.options());
  auto rbf = torch::empty({n_edges, n_radial}, positions.options());
  auto env = torch::empty({n_edges, 1}, positions.options());
  auto status = torch::zeros({}, positions.options().dtype(torch::kInt32).requires_grad(false));
  if (n_edges == 0) return {vec, distance, unit, rbf, env, status};

  const int64_t n_tasks = static_cast<int64_t>(opcodes.size());
  std::vector<int64_t> task_meta;
  std::vector<int64_t> dependent_offsets{0};
  std::vector<int64_t> dependent_ids;
  std::vector<std::vector<int64_t>> children(n_tasks);
  std::vector<int64_t> parent_counts(n_tasks, 0);
  std::vector<int64_t> roots;
  int64_t total_tickets = 0;
  for (int64_t child = 0; child < n_tasks; ++child) {
    for (int64_t j = dependency_offsets[child]; j < dependency_offsets[child + 1]; ++j) {
      children[dependency_ids[j]].push_back(child);
      ++parent_counts[child];
    }
  }
  for (int64_t task = 0; task < n_tasks; ++task) {
    const int64_t tiles = n_edges / tile_sizes[task] + (n_edges % tile_sizes[task] != 0);
    const int64_t dependent_begin = static_cast<int64_t>(dependent_ids.size());
    dependent_ids.insert(dependent_ids.end(), children[task].begin(), children[task].end());
    dependent_offsets.push_back(static_cast<int64_t>(dependent_ids.size()));
    task_meta.insert(task_meta.end(), {
        opcodes[task], work_items[task], tile_sizes[task], tiles,
        parent_counts[task], dependent_begin,
        static_cast<int64_t>(children[task].size()), epochs[task]});
    if (parent_counts[task] == 0) {
      for (int64_t tile = 0; tile < tiles; ++tile) {
        roots.push_back(task);
        roots.push_back(tile);
      }
    }
    total_tickets += tiles;
  }
  TORCH_CHECK(total_tickets == queue_capacity, "runtime tile count disagrees with the submitted plan");

  auto i64_options = positions.options().dtype(torch::kInt64).requires_grad(false);
  auto meta = torch::tensor(task_meta, i64_options).reshape({n_tasks, kMetaWidth});
  auto roots_tensor = torch::tensor(roots, i64_options).reshape({static_cast<int64_t>(roots.size() / 2), 2});
  auto dependent_offsets_tensor = torch::tensor(dependent_offsets, i64_options);
  auto dependent_ids_tensor = torch::tensor(dependent_ids, i64_options);
  auto queue = torch::zeros({queue_capacity, kTicketWidth}, i64_options);
  auto states = torch::empty({n_tasks, 3}, i64_options);
  auto control = torch::zeros({4}, i64_options);
  auto args = dcu::megakernel::EdgeFeaturesArgs{
      positions.data_ptr<float>(), edge_index.data_ptr<int64_t>(), shifts.data_ptr<int64_t>(),
      cells.data_ptr<float>(), edge_frame.data_ptr<int64_t>(), mask.data_ptr<bool>(),
      frequencies.data_ptr<float>(), n_edges, n_radial, float(cutoff),
      int(basis_exponent), int(edge_exponent), float(eps), vec.data_ptr<float>(),
      distance.data_ptr<float>(), unit.data_ptr<float>(), rbf.data_ptr<float>(), env.data_ptr<float>()};
  auto stream = at::cuda::getCurrentCUDAStream(positions.get_device());
  const int64_t init_work = std::max(n_tasks, roots_tensor.size(0));
  hipLaunchKernelGGL(initialize_runtime_kernel,
      dim3(blocks(init_work)), dim3(256), 0, stream,
      meta.data_ptr<int64_t>(), n_tasks, roots_tensor.data_ptr<int64_t>(), roots_tensor.size(0),
      total_tickets, generation, queue.data_ptr<int64_t>(), queue_capacity,
      states.data_ptr<int64_t>(), control.data_ptr<int64_t>(), status.data_ptr<int32_t>());
  launch_check();

  hipDeviceProp_t properties{};
  TORCH_CHECK(hipGetDeviceProperties(&properties, positions.get_device()) == hipSuccess,
              "failed to query HIP device properties");
  // At most one persistent CTA per compute unit keeps the worker pool resident.
  const unsigned worker_blocks = static_cast<unsigned>(
      std::min<int64_t>(total_tickets, std::max(1, properties.multiProcessorCount)));
  hipLaunchKernelGGL(persistent_edge_features_worker_kernel,
      dim3(worker_blocks), dim3(256), 0, stream, args, meta.data_ptr<int64_t>(), n_tasks,
      dependent_offsets_tensor.data_ptr<int64_t>(), dependent_ids_tensor.data_ptr<int64_t>(),
      queue.data_ptr<int64_t>(), queue_capacity, states.data_ptr<int64_t>(),
      control.data_ptr<int64_t>(), status.data_ptr<int32_t>(), generation);
  launch_check();
  return {vec, distance, unit, rbf, env, status};
}
