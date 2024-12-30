#pragma once

#include <span>

#include "ParallelExecutor.hpp"

template <typename ValueType, int n_var>
struct ExponentialODE {
  void operator()(auto& exe, std::span<ValueType const, n_var> x,
                  std::span<ValueType, n_var> dxdt) {
    call_parallel_kernel<exp_ode_kernel>(exe, n_var, x, dxdt);
  }

  static constexpr auto exp_ode_kernel(int i,
                                       std::span<ValueType const, n_var> x,
                                       std::span<ValueType, n_var> dxdt) {
    dxdt[i] = x[i];
  }
};
