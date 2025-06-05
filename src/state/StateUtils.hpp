#pragma once

#include "ParallelExecutor.hpp"

template <typename ValueType>
constexpr void fill_kernel(int i, ValueType* state, ValueType value) {
  state[i] = value;
}

template <typename ParallelExecutor,
          template <template <typename, std::size_t> typename, typename,
                    std::size_t> typename StateAllocator,
          template <typename, std::size_t> typename StateType,
          typename ValueType, std::size_t N>
void fill(ParallelExecutor& exe, StateAllocator<StateType, ValueType, N>& state,
          ValueType value) {
  call_parallel_kernel<fill_kernel<ValueType>>(exe, N, state.data(), value);
}
