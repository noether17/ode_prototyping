#pragma once

#include <array>

#include "HeapState.hpp"
#include "RawOutput.hpp"

// Defines test parameters for Van Der Pol test.
template <template <typename, int> typename StateType>
struct VanDerPolTest {
  auto static constexpr n_var = 2;
  auto static inline const x0 = StateType<double, n_var>(std::array{2.0, 0.0});
  auto static constexpr t0 = 0.0;
  auto static constexpr tf = 2.0;
  auto static constexpr tol = 1.0e-10;
  auto static inline const atol =
      StateType<double, n_var>(std::array{tol, tol});
  auto static inline const rtol = atol;
  auto static constexpr ode_kernel(int i, double const* x, double* dxdt) {
    if (i == 0) {
      dxdt[0] = x[1];
    } else if (i == 1) {
      auto constexpr eps = 1.0;
      dxdt[1] = eps * (1.0 - x[0] * x[0]) * x[1] - x[0];
    }
  }
  auto constexpr operator()(auto& exe, auto const& x, auto* dxdt) {
    exe.template call_parallel_kernel<ode_kernel>(n_var, x.data(), dxdt);
  }
  RawOutput<HeapState<double, n_var>> output{};
};
