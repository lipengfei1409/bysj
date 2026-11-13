// Standalone CPU Kineto smoke test; no checkpoint, DCU or DeepMD build needed.
#include "PTExptProfiler.h"
#include <ATen/ATen.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

using deepmd::ptexpt::WindowProfiler;

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string base = argv[1];
  unsetenv("DP_PROFILER");
  setenv("DP_PROFILER_SKIP", "invalid", 1);
  {
    auto disabled = WindowProfiler::from_environment(false, 0, [] {});
    WindowProfiler::Scope scope(disabled.get());
    at::ones({4}).sum();
  }
  setenv("DP_PROFILER", base.c_str(), 1);
  bool rejected = false;
  try {
    auto invalid = WindowProfiler::from_environment(false, 0, [] {});
  } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  setenv("DP_PROFILER_SKIP", "2", 1);
  setenv("DP_PROFILER_ACTIVE", "3", 1);
  int syncs = 0;
  {
    auto profiler = WindowProfiler::from_environment(false, 0, [&] { ++syncs; });
    for (int call = 0; call < 7; ++call) {
      WindowProfiler::Scope outer(profiler.get());
      WindowProfiler::Scope inner(profiler.get());  // nesting counts once
      auto x = at::ones({4});
      x.sum();
    }
  }
  assert(syncs == 2);
  setenv("DP_PROFILER", (base + "_partial").c_str(), 1);
  {
    auto profiler = WindowProfiler::from_environment(false, 0, [] {});
    for (int call = 0; call < 3; ++call) {
      WindowProfiler::Scope scope(profiler.get());
      at::ones({4}).sum();
    }
  }
  setenv("DP_PROFILER_SKIP", "0", 1);
  setenv("DP_PROFILER", (base + "_aborted").c_str(), 1);
  {
    auto profiler = WindowProfiler::from_environment(false, 0, [] {});
    try {
      WindowProfiler::Scope scope(profiler.get());
      at::ones({4}).sum();
      throw std::runtime_error("inference failed");
    } catch (const std::runtime_error&) {}
  }
  // Must be able to start another profiler after exception cleanup.
  setenv("DP_PROFILER_ACTIVE", "1", 1);
  setenv("DP_PROFILER", (base + "_after").c_str(), 1);
  {
    auto profiler = WindowProfiler::from_environment(false, 0, [] {});
    WindowProfiler::Scope scope(profiler.get());
    at::ones({4}).sum();
  }
  setenv("DP_PROFILER", (base + "_startup_failure").c_str(), 1);
  {
    int attempts = 0;
    auto profiler = WindowProfiler::from_environment(false, 0, [&] {
      ++attempts;
      throw std::runtime_error("synchronization failed");
    });
    try {
      WindowProfiler::Scope scope(profiler.get());
      assert(false);
    } catch (const std::runtime_error&) {}
    // Failed collection is terminal, but a retried inference remains safe.
    WindowProfiler::Scope retry(profiler.get());
    at::ones({4}).sum();
    assert(attempts == 1);
  }

}
