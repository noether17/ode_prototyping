#pragma once

#include "ParallelExecutor.hpp"

template <int n_var>
struct ExponentialODE {
  void operator()(auto& exe, auto const& x, auto* dxdt) {
    call_parallel_kernel<exp_ode_kernel>(exe, n_var, x.data(), dxdt);
  }

  static constexpr auto exp_ode_kernel(int i, double const* x, double* dxdt) {
    dxdt[i] = x[i];
  }
};
