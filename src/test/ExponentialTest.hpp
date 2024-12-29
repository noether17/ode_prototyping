#pragma once

#include <array>
#include <numeric>

#include "ExponentialODE.hpp"
#include "HeapState.hpp"
#include "RawOutput.hpp"

// Defines parameters for Exponential test.
template <template <typename, int> typename StateType>
struct ExponentialTest {
  static constexpr auto n_var = 10;
  static constexpr auto x0_data = [] {
    auto temp = std::array<double, n_var>{};
    std::iota(temp.begin(), temp.end(), 0.0);
    return temp;
  }();
  auto static inline const x0 = StateType<double, n_var>(x0_data);
  static constexpr auto t0 = 0.0;
  static constexpr auto tf = 10.0;
  static constexpr auto tol = 1.0e-6;
  static constexpr auto tol_array = [] {
    auto temp = std::array<double, n_var>{};
    std::fill(temp.begin(), temp.end(), tol);
    return temp;
  }();
  auto static inline const atol = StateType<double, n_var>{tol_array};
  auto static inline const rtol = atol;
  constexpr auto operator()(auto& exe, auto const& x, auto* dxdt) {
    ExponentialODE<n_var>{}(exe, x, dxdt);
  }
  RawOutput<HeapState<double, n_var>> output{};
};
