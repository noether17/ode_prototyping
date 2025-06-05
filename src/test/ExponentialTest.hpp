#pragma once

#include <array>
#include <numeric>
#include <span>

#include "ExponentialODE.hpp"
#include "HeapState.hpp"
#include "RawOutput.hpp"

// Defines parameters for Exponential test.
template <template <template <typename, std::size_t> typename, typename,
                    std::size_t> typename StateType>
struct ExponentialTest {
  static constexpr auto n_var = 10;
  static constexpr auto x0_data = [] {
    auto temp = std::array<double, n_var>{};
    std::iota(temp.begin(), temp.end(), 0.0);
    return temp;
  }();
  static inline auto const x0 = StateType{x0_data};
  static constexpr auto t0 = 0.0;
  static constexpr auto tf = 10.0;
  static constexpr auto tol = 1.0e-6;
  static constexpr auto tol_array = [] {
    auto temp = std::array<double, n_var>{};
    std::fill(temp.begin(), temp.end(), tol);
    return temp;
  }();
  static inline auto const atol = StateType{tol_array};
  static inline auto const rtol = atol;
  constexpr auto operator()(auto& exe, std::span<double const, n_var> x,
                            std::span<double, n_var> dxdt) {
    ExponentialODE<double, n_var>{}(exe, x, dxdt);
  }
  RawOutput<HeapState<std::array, double, n_var>> output{};
};
