#pragma once

// TODO: This library should be replaced with the existing host utilities.

#include <cmath>
#include <numeric>
#include <vector>

template <typename RKMethod>
auto host_estimate_initial_step(std::vector<double> const& x0,
                                std::vector<double> const& atol,
                                std::vector<double> const& rtol) -> double {
  auto error_target = std::vector<double>(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    error_target[i] = atol[i] + rtol[i] * std::abs(x0[i]);
  }

  auto f0 = x0;  // exponential ODE just copies state to rhs
  auto d0 = std::sqrt(
      std::inner_product(x0.begin(), x0.end(), error_target.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      x0.size());
  auto d1 = std::sqrt(
      std::inner_product(f0.begin(), f0.end(), error_target.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      x0.size());
  auto dt0 = (d0 < 1.0e-5 or d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (d0 / d1);

  auto x1 = std::vector<double>(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    x1[i] = x0[i] + f0[i] * dt0;
  }
  auto f1 = x1;  // exponential ODE just copies state to rhs
  auto df = std::vector<double>(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    df[i] = f1[i] - f0[i];
  }
  auto d2 = std::sqrt(std::inner_product(
                          df.begin(), df.end(), error_target.begin(), 0.0,
                          std::plus<>{},
                          [](auto a, auto b) { return (a / b) * (a / b); }) /
                      x0.size()) /
            dt0;

  auto constexpr p = RKMethod::p;
  auto dt1 = (std::max(d1, d2) <= 1.0e-15)
                 ? std::max(1.0e-6, dt0 * 1.0e-3)
                 : std::pow(0.01 / std::max(d1, d2), (1.0 / (1.0 + p)));
  return std::min(100.0 * dt0, dt1);
}

template <int n_var, typename RKMethod, typename ODE>
void host_evaluate_stages(double const* x0, double* temp_state, double* ks,
                          double dt) {
  ODE::compute_rhs(x0, ks);
  for (auto stage = 1; stage < RKMethod::n_stages; ++stage) {
    std::fill(temp_state, temp_state + n_var, 0.0);
    for (auto i = 0; i < n_var; ++i) {
      temp_state[i] = x0[i];
      for (auto j = 0; j < stage; ++j) {
        temp_state[i] += RKMethod::a[stage - 1][j] * ks[j * n_var + i] * dt;
      }
    }
    ODE::compute_rhs(temp_state, ks + stage * n_var);
  }
}

template <int n_var>
struct HostExpODE {
  static void compute_rhs(double const* x, double* f) {
    std::copy(x, x + n_var, f);
  }
};
