#pragma once

#include <array>

#include "HeapState.hpp"
#include "RawOutput.hpp"
#include "VanDerPolODE.hpp"

// Defines test parameters for Van Der Pol test.
template <template <typename, int> typename StateType>
struct VanDerPolTest {
  static constexpr auto n_var = 2;
  auto static inline const x0 = StateType<double, n_var>(std::array{2.0, 0.0});
  static constexpr auto t0 = 0.0;
  static constexpr auto tf = 2.0;
  static constexpr auto tol = 1.0e-10;
  auto static inline const atol =
      StateType<double, n_var>(std::array{tol, tol});
  auto static inline const rtol = atol;
  constexpr auto operator()(auto& exe, auto const& x, auto* dxdt) {
    VanDerPolODE{}(exe, x, dxdt);
  }
  RawOutput<HeapState<double, n_var>> output{};
};
