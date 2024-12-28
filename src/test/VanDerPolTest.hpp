#pragma once

#include <array>

#include "HeapState.hpp"
#include "RawOutput.hpp"
#include "VanDerPolODE.hpp"

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
  auto constexpr operator()(auto& exe, auto const& x, auto* dxdt) {
    VanDerPolODE{}(exe, x, dxdt);
  }
  RawOutput<HeapState<double, n_var>> output{};
};
