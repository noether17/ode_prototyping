#pragma once

#include "ParallelExecutor.hpp"

template <typename ValueType>
constexpr void fill_kernel(int i, ValueType* state, ValueType value) {
  state[i] = value;
}

template <typename ParallelExecutor,
          template <typename, std::size_t> typename StateAllocator,
          typename ValueType, std::size_t N>
void fill(ParallelExecutor& exe, StateAllocator<ValueType, N>& state,
          ValueType value) {
  call_parallel_kernel<fill_kernel<ValueType>>(exe, N, state.data(), value);
}
