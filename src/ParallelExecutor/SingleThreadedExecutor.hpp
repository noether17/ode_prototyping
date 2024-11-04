#pragma once

#include <utility>

struct SingleThreadedExecutor {
  template <typename ParallelKernel, typename... Args>
  void call_parallel_kernel(ParallelKernel kernel, int n_items,
                            Args&&... args) {
    for (auto i = 0; i < n_items; ++i) {
      kernel(i, std::forward<Args>(args)...);
    }
  }

  // TODO: This is needed temporarily for compatibility with
  // RKEmbeddedParallel's expectation of ParallelThreadPool. Need to modify the
  // interface of ParallelThreadPool to allow caller to request a reduction
  // without needing n_threads().
  auto static constexpr n_threads() { return 1; }
};
