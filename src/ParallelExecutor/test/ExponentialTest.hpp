#pragma once

#include <array>
#include <numeric>

#include "ExponentialODE.hpp"
#include "HeapState.hpp"
#include "RawOutput.hpp"

// Defines parameters for Exponential test.
template <template <typename, int> typename StateType>
struct ExponentialTest {
  auto static constexpr n_var = 10;
  auto static constexpr x0_data = [] {
    auto temp = std::array<double, n_var>{};
    std::iota(temp.begin(), temp.end(), 0.0);
    return temp;
  }();
  auto static inline const x0 = StateType<double, n_var>(x0_data);
  auto static constexpr t0 = 0.0;
  auto static constexpr tf = 10.0;
  auto static constexpr tol = 1.0e-6;
  auto static constexpr tol_array = [] {
    auto temp = std::array<double, n_var>{};
    std::fill(temp.begin(), temp.end(), tol);
    return temp;
  }();
  auto static inline const atol = StateType<double, n_var>{tol_array};
  auto static inline const rtol = atol;
  auto constexpr operator()(auto& exe, auto const& x, auto* dxdt) {
    exe.template call_parallel_kernel<exp_ode_kernel>(n_var, x.data(), dxdt);
  }
  RawOutput<HeapState<double, n_var>> output{};
};
