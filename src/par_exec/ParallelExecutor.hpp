#pragma once

#include "KernelConcepts.hpp"

template <auto kernel, typename ParallelExecutor, typename... Args>
void call_parallel_kernel(ParallelExecutor& exe, int n_items, Args... args)
  requires ParallelKernel<kernel, Args...>
{
  exe.template call_parallel_kernel<kernel>(n_items, args...);
}

template <typename T, auto reduce, auto transform, typename ParallelExecutor,
          typename... TransformArgs>
T transform_reduce(ParallelExecutor& exe, T init_val, int n_items,
                   TransformArgs... transform_args)
  requires(TransformKernel<transform, T, TransformArgs...> and
           ReductionOp<reduce, T>)
{
  return exe.template transform_reduce<T, reduce, transform>(init_val, n_items,
                                                             transform_args...);
}
