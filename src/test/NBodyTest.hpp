#pragma once

#include <array>
#include <span>

#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "RawOutput.hpp"

// Defines parameters for NBody test.
template <template <template <typename, std::size_t> typename, typename,
                    std::size_t> typename StateType>
struct NBodyTest {
  static constexpr auto x0_data =
      std::array{1.657666,  0.0,       0.0, 0.439775,  -0.169717, 0.0,
                 -1.268608, -0.267651, 0.0, -1.268608, 0.267651,  0.0,
                 0.439775,  0.169717,  0.0, 0.0,       -0.593786, 0.0,
                 1.822785,  0.128248,  0.0, 1.271564,  0.168645,  0.0,
                 -1.271564, 0.168645,  0.0, -1.822785, 0.128248,  0.0};
  static constexpr auto masses = std::array{1.0, 1.0, 1.0, 1.0, 1.0};
  static constexpr auto n_var = x0_data.size();
  static inline auto const x0 = StateType{x0_data};
  static constexpr auto t0 = 0.0;
  static constexpr auto tf = 6.3;
  static constexpr auto tol = 1.0e-10;
  static constexpr auto tol_array = [] {
    auto temp = std::array<double, n_var>{};
    std::fill(temp.begin(), temp.end(), tol);
    return temp;
  }();
  static inline auto const atol = StateType{tol_array};
  static inline auto const rtol = atol;
  constexpr auto operator()(auto& exe, std::span<double const, n_var> x,
                            std::span<double, n_var> dxdt) {
    NBodyODE<double, n_var>{}(exe, x, dxdt);
  }
  RawOutput<HeapState<std::array, double, n_var>> output{};
};
