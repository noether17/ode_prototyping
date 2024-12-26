#pragma once

#include <utility>

struct SingleThreadedExecutor {
  template <auto parallel_kernel, typename... Args>
  void call_parallel_kernel(int n_items, Args... args) {
    for (auto i = 0; i < n_items; ++i) {
      parallel_kernel(i, std::move(args)...);
    }
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  auto transform_reduce(T init_val, int n_items,
                        TransformArgs... transform_args) {
    auto result = init_val;
    for (auto i = 0; i < n_items; ++i) {
      auto transform_result = transform(i, std::move(transform_args)...);
      result = reduce(result, transform_result);
    }
    return result;
  }
};
