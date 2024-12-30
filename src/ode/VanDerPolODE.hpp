#pragma once

#include <span>

#include "ParallelExecutor.hpp"

template <typename ValueType>
struct VanDerPolODE {
  static constexpr auto n_var = 2;

  void operator()(auto& exe, std::span<ValueType const, n_var> x,
                  std::span<ValueType, n_var> dxdt) {
    call_parallel_kernel<VDP_ode_kernel>(exe, n_var, x, dxdt);
  }

  static constexpr void VDP_ode_kernel(int i,
                                       std::span<ValueType const, n_var> x,
                                       std::span<ValueType, n_var> dxdt) {
    if (i == 0) {
      dxdt[0] = x[1];
    } else if (i == 1) {
      constexpr auto eps = 1.0;
      dxdt[1] = eps * (1.0 - x[0] * x[0]) * x[1] - x[0];
    }
  }
};
