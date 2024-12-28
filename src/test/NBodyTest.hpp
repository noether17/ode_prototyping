#pragma once

#include <array>

#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "RawOutput.hpp"

// Defines parameters for Exponential test.
template <template <typename, int> typename StateType>
struct NBodyTest {
  auto static constexpr x0_data =
      std::array{1.657666,  0.0,       0.0, 0.439775,  -0.169717, 0.0,
                 -1.268608, -0.267651, 0.0, -1.268608, 0.267651,  0.0,
                 0.439775,  0.169717,  0.0, 0.0,       -0.593786, 0.0,
                 1.822785,  0.128248,  0.0, 1.271564,  0.168645,  0.0,
                 -1.271564, 0.168645,  0.0, -1.822785, 0.128248,  0.0};
  auto static constexpr masses = std::array{1.0, 1.0, 1.0, 1.0, 1.0};
  auto static constexpr n_var = x0_data.size();
  auto static inline const x0 = StateType<double, n_var>(x0_data);
  auto static constexpr t0 = 0.0;
  auto static constexpr tf = 6.3;
  auto static constexpr tol = 1.0e-10;
  auto static constexpr tol_array = [] {
    auto temp = std::array<double, n_var>{};
    std::fill(temp.begin(), temp.end(), tol);
    return temp;
  }();
  auto static inline const atol = StateType<double, n_var>{tol_array};
  auto static inline const rtol = atol;
  auto constexpr operator()(auto& exe, auto const& x, auto* dxdt) {
    NBodyODE<n_var>{}(exe, x, dxdt);
  }
  RawOutput<HeapState<double, n_var>> output{};
};
