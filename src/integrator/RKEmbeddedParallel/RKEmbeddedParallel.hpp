#pragma once

#include <array>
#include <cmath>
#include <span>
#include <vector>

#include "ParallelExecutor.hpp"

template <template <typename, int> typename StateContainer, typename ValueType,
          int NVAR, typename ButcherTableau, typename ODE, typename Output,
          typename ParallelExecutor>
struct RKEmbeddedParallel {
  void integrate(StateContainer<ValueType, NVAR> x0, double t0, double tf,
                 StateContainer<ValueType, NVAR> atol,
                 StateContainer<ValueType, NVAR> rtol, ODE ode, Output& output,
                 ParallelExecutor& exe) {
    static constexpr auto max_step_scale = 6.0;
    static constexpr auto min_step_scale = 0.33;
    static constexpr auto q = std::min(ButcherTableau::p, ButcherTableau::pt);
    static auto const safety_factor = std::pow(0.38, (1.0 / (1.0 + q)));
    auto ks = StateContainer<ValueType, ButcherTableau::n_stages * NVAR>{};

    auto dt = detail::estimate_initial_step(exe, x0, atol, rtol, ode);
    auto t = t0;
    auto x = x0;
    auto n_var = std::ssize(x);
    auto temp_state = StateContainer<ValueType, NVAR>{};
    auto error_estimate = StateContainer<ValueType, NVAR>{};
    auto error_target = StateContainer<ValueType, NVAR>{};
    output.save_state(t, x);
    while (t < tf) {
      // evaluate stages
      ode(exe, x0,
          std::span<ValueType, ButcherTableau::n_stages * NVAR>{ks}
              .template subspan<0, NVAR>());
      for (auto stage = 1; stage < ButcherTableau::n_stages; ++stage) {
        call_parallel_kernel<detail::rk_stage_kernel>(
            exe, n_var, stage, dt, temp_state.data(), ks.data(), x0.data());
        ode(exe, temp_state,
            std::span<ValueType, NVAR>(ks.data() + stage * NVAR, NVAR));
      }

      // advance the state and compute the error estimate
      call_parallel_kernel<detail::update_state_and_error_kernel>(
          exe, n_var, dt, x.data(), error_estimate.data(), ks.data(),
          x0.data());

      // estimate error
      call_parallel_kernel<detail::update_error_target_kernel>(
          exe, n_var, error_target.data(), atol.data(), rtol.data(), x.data(),
          x0.data());
      auto scaled_error = detail::rk_norm(exe, error_estimate, error_target);

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

  struct detail {
    static constexpr auto rk_stage_kernel(int i, int stage, ValueType dt,
                                          ValueType* temp_state, ValueType* ks,
                                          ValueType const* x0) {
      constexpr auto a = ButcherTableau::a;
      temp_state[i] = 0.0;
      for (auto j = 0; j < stage; ++j) {
        temp_state[i] += a[stage - 1][j] * ks[j * NVAR + i];
      }
      temp_state[i] = x0[i] + temp_state[i] * dt;
    }

    static constexpr auto update_state_and_error_kernel(
        int i, ValueType dt, ValueType* x, ValueType* error_estimate,
        ValueType* ks, ValueType const* x0) {
      constexpr auto b = ButcherTableau::b;
      constexpr auto db = [] {
        auto db = ButcherTableau::b;
        for (auto i = 0; i < ButcherTableau::n_stages; ++i) {
          db[i] -= ButcherTableau::bt[i];
        }
        return db;
      }();
      x[i] = 0.0;
      error_estimate[i] = 0.0;
      for (auto j = 0; j < ButcherTableau::n_stages; ++j) {
        x[i] += b[j] * ks[j * NVAR + i];
        error_estimate[i] += db[j] * ks[j * NVAR + i];
      }
      x[i] = x0[i] + x[i] * dt;
      error_estimate[i] *= dt;
    }

    static constexpr auto update_error_target_kernel(
        int i, ValueType* error_target, ValueType const* atol,
        ValueType const* rtol, ValueType const* x, ValueType const* x0) {
      error_target[i] =
          atol[i] + rtol[i] * std::max(std::abs(x[i]), std::abs(x0[i]));
    }

    static constexpr auto scaled_value_squared_kernel(int i, ValueType const* v,
                                                      ValueType const* scale) {
      auto scaled_value = v[i] / scale[i];
      return scaled_value * scaled_value;
    }

    static constexpr auto add(ValueType a, ValueType b) { return a + b; }

    ValueType static rk_norm(ParallelExecutor& exe,
                             std::span<ValueType const, NVAR> v,
                             std::span<ValueType const, NVAR> scale) {
      auto n_var = std::ssize(v);
      return std::sqrt(
          transform_reduce<ValueType, add, scaled_value_squared_kernel>(
              exe, 0.0, n_var, v.data(), scale.data()) /
          n_var);
    }

    static constexpr auto compute_error_target_kernel(int i,
                                                      ValueType* error_target,
                                                      ValueType const* x0,
                                                      ValueType const* atol,
                                                      ValueType const* rtol) {
      error_target[i] = std::abs(x0[i]) * rtol[i] + atol[i];
    }

    static constexpr auto euler_step_kernel(int i, ValueType* x1,
                                            ValueType const* x0,
                                            ValueType const* f0,
                                            ValueType dt0) {
      x1[i] = x0[i] + f0[i] * dt0;
    }

    static constexpr auto difference_kernel(int i, ValueType* df,
                                            ValueType const* f0) {
      df[i] -= f0[i];
    }

    double static estimate_initial_step(ParallelExecutor& exe,
                                        std::span<ValueType const, NVAR> x0,
                                        std::span<ValueType const, NVAR> atol,
                                        std::span<ValueType const, NVAR> rtol,
                                        ODE& ode) {
      auto const n_var = std::ssize(x0);

      auto error_target = StateContainer<ValueType, NVAR>{};
      call_parallel_kernel<compute_error_target_kernel>(
          exe, n_var, error_target.data(), x0.data(), atol.data(), rtol.data());

      auto f0 = StateContainer<ValueType, NVAR>{};
      ode(exe, x0, f0);
      auto d0 = rk_norm(exe, x0, error_target);
      auto d1 = rk_norm(exe, f0, error_target);
      auto dt0 = (d0 < 1.0e-5 or d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (d0 / d1);

      auto x1 = StateContainer<ValueType, NVAR>{};
      call_parallel_kernel<euler_step_kernel>(exe, n_var, x1.data(), x0.data(),
                                              f0.data(), dt0);
      auto df = StateContainer<ValueType, NVAR>{};
      ode(exe, x1, df);
      call_parallel_kernel<difference_kernel>(exe, n_var, df.data(), f0.data());
      auto d2 = rk_norm(exe, df, error_target) / dt0;

      constexpr auto p = ButcherTableau::p;
      auto dt1 = (std::max(d1, d2) <= 1.0e-15)
                     ? std::max(1.0e-6, dt0 * 1.0e-3)
                     : std::pow(0.01 / std::max(d1, d2), (1.0 / (1.0 + p)));
      return std::min(100.0 * dt0, dt1);
    }
  };
};
