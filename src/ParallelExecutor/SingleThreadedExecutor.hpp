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

  template <typename T, typename BinaryOp, typename TransformOp,
            typename... Args>
  auto transform_reduce(T init_val, BinaryOp reduce, TransformOp transform,
                        int n_items, Args&&... transform_args) {
    auto result = init_val;
    for (auto i = 0; i < n_items; ++i) {
      auto transform_result =
          transform(i, std::forward<Args>(transform_args)...);
      result = reduce(result, transform_result);
    }
    return result;
  }
};
