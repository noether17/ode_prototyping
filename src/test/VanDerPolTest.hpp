#pragma once

#include <array>

#include "HeapState.hpp"
#include "RawOutput.hpp"
#include "VanDerPolODE.hpp"

// Defines test parameters for Van Der Pol test.
template <template <template <typename, std::size_t> typename, typename,
                    std::size_t> typename StateType>
struct VanDerPolTest {
  static constexpr auto n_var = 2;
  static inline auto const x0 = StateType{std::array{2.0, 0.0}};
  static constexpr auto t0 = 0.0;
  static constexpr auto tf = 2.0;
  static constexpr auto tol = 1.0e-10;
  static inline auto const atol = StateType{std::array{tol, tol}};
  static inline auto const rtol = atol;
  constexpr auto operator()(auto& exe, std::span<double const, n_var> x,
                            std::span<double, n_var> dxdt) {
    VanDerPolODE<double>{}(exe, x, dxdt);
  }
  RawOutput<HeapState<std::array, double, n_var>> output{};
};
