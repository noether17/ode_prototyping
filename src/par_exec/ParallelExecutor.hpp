#pragma once

#include <utility>

#include "KernelConcepts.hpp"

template <auto kernel, typename ParallelExecutor, typename... Args>
void call_parallel_kernel(ParallelExecutor& exe, std::size_t n_items,
                          Args... args)
  requires ParallelKernel<kernel, Args...>
{
  exe.template call_parallel_kernel<kernel>(n_items, std::move(args)...);
}

template <typename T, auto reduce, auto transform, typename ParallelExecutor,
          typename... TransformArgs>
T transform_reduce(ParallelExecutor& exe, T init_val, std::size_t n_items,
                   TransformArgs... transform_args)
  requires(TransformKernel<transform, T, TransformArgs...> and
           ReductionOp<reduce, T>)
{
  return exe.template transform_reduce<T, reduce, transform>(
      init_val, n_items, std::move(transform_args)...);
}
