#pragma once

#include <array>
#include <cmath>
#include <numeric>

#include "ParallelExecutor.hpp"

template <typename StateType, typename ButcherTableau, typename ODE,
          typename Output>
class RKEmbeddedParallel {
 public:
  auto integrate(StateType x0, double t0, double tf, StateType atol,
                 StateType rtol, ODE ode, Output& output,
                 ParallelExecutor& exe) -> void {
    auto static constexpr max_step_scale = 6.0;
    auto static constexpr min_step_scale = 0.33;
    auto static constexpr db = [] {
      auto db = ButcherTableau::b;
      for (auto i = 0; i < ButcherTableau::n_stages; ++i) {
        db[i] -= ButcherTableau::bt[i];
      }
      return db;
    }();
    auto constexpr q = std::min(ButcherTableau::p, ButcherTableau::pt);
    auto static const safety_factor = std::pow(0.38, (1.0 / (1.0 + q)));
    auto constexpr a = ButcherTableau::a;
    auto constexpr b = ButcherTableau::b;
    auto constexpr n_stages = ButcherTableau::n_stages;
    auto ks = std::array<StateType, n_stages>{};

    auto dt = estimate_initial_step(exe, x0, atol, rtol, ode);
    auto t = t0;
    auto x = x0;
    auto n_var = std::ssize(x);
    auto temp_state = StateType{};
    auto error_estimate = StateType{};
    auto error_target = StateType{};
    output.save_state(t, x);
    while (t < tf) {
      // evaluate stages
      ode(x0, ks[0]);
      for (auto stage = 1; stage < n_stages; ++stage) {
        exe.call_parallel_kernel(
            [&, stage, dt](int i) {
              temp_state[i] = 0.0;
              for (auto j = 0; j < stage; ++j) {
                temp_state[i] += a[stage - 1][j] * ks[j][i];
              }
              temp_state[i] = x0[i] + temp_state[i] * dt;
            },
            n_var);
        ode(temp_state, ks[stage]);
      }

      // advance the state and compute the error estimate
      exe.call_parallel_kernel(
          [&, n_stages, dt](int i) {
            x[i] = 0.0;
            error_estimate[i] = 0.0;
            for (auto j = 0; j < n_stages; ++j) {
              x[i] += b[j] * ks[j][i];
              error_estimate[i] += db[j] * ks[j][i];
            }
            x[i] = x0[i] + x[i] * dt;
            error_estimate[i] *= dt;
          },
          n_var);

      // estimate error
      exe.call_parallel_kernel(
          [&](int i) {
            error_target[i] =
                atol[i] + rtol[i] * std::max(std::abs(x[i]), std::abs(x0[i]));
          },
          n_var);
      auto scaled_error = rk_norm(exe, error_estimate, error_target);

      // accept or reject the step
      if (scaled_error <= 1.0) {
        t += dt;
        x0 = x;
        output.save_state(t, x);
      }

      // update step size
      auto dtnew = dt * safety_factor / std::pow(scaled_error, 1.0 / (1.0 + q));
      if (std::abs(dtnew) > max_step_scale * std::abs(dt)) {
        dt *= max_step_scale;
      } else if (std::abs(dtnew) < min_step_scale * std::abs(dt)) {
        dt *= min_step_scale;
      } else if (dtnew / (tf - t0) < 1.0e-12) {
        dt = (tf - t0) * 1.0e-12;
      } else {
        dt = dtnew;
      }

      // detect last step and correct overshooting
      if (t + dt > tf) {
        dt = tf - t;
      }
    }
  }

  auto static rk_norm(ParallelExecutor& exe, StateType const& v,
                      StateType const& scale) {
    auto const n_threads = exe.n_threads();
    auto const n_var = std::ssize(v);
    auto thread_partial_results = std::vector<double>(n_threads);
    auto n_items_per_thread = (n_var + n_threads - 1) / n_threads;
    exe.call_parallel_kernel(
        [&](int thread_id) {
          auto thread_partial_result = 0.0;
          for (auto i = thread_id * n_items_per_thread;
               i < (thread_id + 1) * n_items_per_thread and i < n_var; ++i) {
            auto scaled_value = v[i] / scale[i];
            thread_partial_result += scaled_value * scaled_value;
          }
          thread_partial_results[thread_id] = thread_partial_result;
        },
        n_threads);
    return std::sqrt(std::accumulate(thread_partial_results.begin(),
                                     thread_partial_results.end(), 0.0) /
                     n_var);
  }

  auto static estimate_initial_step(ParallelExecutor& exe, StateType const& x0,
                                    StateType const& atol,
                                    StateType const& rtol, ODE& ode) {
    auto const n_var = std::ssize(x0);

    auto error_target = StateType{};
    exe.call_parallel_kernel(
        [&](int i) { error_target[i] = atol[i] + rtol[i] * std::abs(x0[i]); },
        n_var);

    auto f0 = StateType{};
    ode(x0, f0);
    auto d0 = rk_norm(exe, x0, error_target);
    auto d1 = rk_norm(exe, f0, error_target);
    auto dt0 = (d0 < 1.0e-5 or d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (d0 / d1);

    auto x1 = StateType{};
    exe.call_parallel_kernel([&](int i) { x1[i] = x0[i] + f0[i] * dt0; },
                             n_var);
    auto df = StateType{};
    ode(x1, df);
    exe.call_parallel_kernel([&](int i) { df[i] -= f0[i]; }, n_var);
    auto d2 = rk_norm(exe, df, error_target) / dt0;

    auto constexpr p = ButcherTableau::p;
    auto dt1 = (std::max(d1, d2) <= 1.0e-15)
                   ? std::max(1.0e-6, dt0 * 1.0e-3)
                   : std::pow(0.01 / std::max(d1, d2), (1.0 / (1.0 + p)));
    return std::min(100.0 * dt0, dt1);
  }
};
