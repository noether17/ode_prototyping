#pragma once

#include <utility>

#include "KernelConcepts.hpp"

struct SingleThreadedExecutor {
  template <auto kernel, typename... Args>
  void call_parallel_kernel(std::size_t n_items, Args... args)
    requires ParallelKernel<kernel, Args...>
  {
    for (std::size_t i = 0; i < n_items; ++i) {
      kernel(i, std::move(args)...);
    }
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  auto transform_reduce(T init_val, std::size_t n_items,
                        TransformArgs... transform_args)
    requires(TransformKernel<transform, T, TransformArgs...> and
             ReductionOp<reduce, T>)
  {
    auto result = init_val;
    for (std::size_t i = 0; i < n_items; ++i) {
      auto transform_result = transform(i, std::move(transform_args)...);
      result = reduce(result, transform_result);
    }
    return result;
  }
};
