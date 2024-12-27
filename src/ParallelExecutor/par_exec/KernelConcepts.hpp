#pragma once

#include <concepts>

/* ParallelKernel concept for constraining callables intended for element-wise
 * operations. Must take an integer index as the first argument, as well as a
 * parameter pack consisting of the data on which to perform the operation. All
 * parameters should be passed by value for CUDA compatibility (to prevent
 * device code from receiving a host address). Any array parameter should be
 * passed using either a raw pointer or a view type such as std::span. */
template <auto parallel_kernel, typename... Args>
concept ParallelKernel = requires(int index, Args... args) {
  { parallel_kernel(index, args...) } -> std::same_as<void>;
};

/* TransformKernel concept for constraining callables intended for element-wise
 * operations immediately preceding a reduction operation. Returns a value to be
 * passed to the reduction operation instead of expecting an output parameter
 * (since an output parameter would likely be a redundant temporary object).
 * Must take an integer index as the first argument, as well as a parameter pack
 * consisting of the data on which to perform the operation. All parameters
 * should be passed by value for CUDA compatibility (to prevent device code from
 * receiving a host address). Any array parameter should be passed using either
 * a raw pointer or a view type such as std::span. */
template <auto transform_kernel, typename T, typename... Args>
concept TransformKernel = requires(int index, Args... args) {
  { transform_kernel(index, args...) } -> std::same_as<T>;
};

/* ReductionOp takes two values and returns a single value. Intended to be
 * called repeatedly to reduce an array of values to a single scalar value. */
template <auto reduction_op, typename T>
concept ReductionOp = requires(T a, T b) {
  { reduction_op(a, b) } -> std::same_as<T>;
};
