#pragma once

#include <array>
#include <cmath>

#include "ParallelExecutor.hpp"

template <typename StateType, typename ButcherTableau, typename ODE,
         typename Output>
class RKEmbeddedParallel {
  auto integrate(StateType x0, double t0, double tf, StateType atol,
      StateType rtol, ODE ode, Output& output, ParallelExecutor& exe) -> void {
    auto static constexpr max_step_scale = 6.0;
    auto static constexpr min_step_scale = 0.33;
    auto static constexpr db = []{
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

    auto dt = estimate_initial_step(x0, atol, rtol, ode);
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
        set_zero(temp_state);
        for (auto j = 0; j < stage; ++j) {
          exe.call_parallel_kernel([](int i, double a, StateType const& ksj, StateType& temp){
              temp[i] += a * ksj[i];
              }, n_var, a[stage - 1][j], ks[j], temp_state);
        }
        exe.call_parallel_kernel([](int i, StateType const& x0, double dt, StateType& temp) {
            temp[i] = x0[i] + temp[i] * dt;
            }, n_var, x0, dt, temp_state);
        ode(temp_state, ks[stage]);
      }

      // advance the state and compute the error estimate
      set_zero(x);
      set_zero(error_estimate);
      exe.call_parallel_kernel([&](int i){
          for (auto j = 0; j < n_stages; ++j) {
          x[i] += b[j] * ks[j][i];
          error_estimate[i] += db[j] * ks[j][i];
          }
          x[i] = x0[i] + x[i] * dt;
          }, n_var);

      // estimate error
      exe.call_parallel_kernel([&](int i) {
          error_target[i] = atol[i] + rtol[i] * std::max(std::abs(x[i]), std::abs(x0[i]));
          }, n_var);
      auto scaled_error = rk_norm(error_estimate, error_target);

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
};
