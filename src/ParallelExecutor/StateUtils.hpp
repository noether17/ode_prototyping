#pragma once

template <typename ValueType>
constexpr void fill_kernel(int i, ValueType* state, ValueType value) {
  state[i] = value;
}

template <typename ParallelExecutor,
          template <typename, int> typename StateType, typename ValueType,
          int N>
void fill(ParallelExecutor& exe, StateType<ValueType, N>& state,
          ValueType value) {
  exe.template call_parallel_kernel<fill_kernel<ValueType>>(N, state.data(),
                                                            value);
}
