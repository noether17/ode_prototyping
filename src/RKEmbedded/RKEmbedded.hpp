#pragma once

#include <cmath>
#include <ranges>

#include "VectorState.hpp"

namespace vws = std::views;

template <typename RKMethod, typename ODE, typename StateType>
class RKEmbedded {
 public:
  auto integrate(StateType x0, double t0, double tf, StateType atol,
                 StateType rtol) -> void {
    auto constexpr a = RKMethod::a;
    auto constexpr b = RKMethod::b;
    auto constexpr n_stages = RKMethod::n_stages;

    auto& ks = derived()->ks;
    auto& ode = derived()->ode;

    auto dt = estimate_initial_step(x0, atol, rtol);
    auto t = t0;
    auto x = x0;
    auto temp_state = StateType{};
    auto error_estimate = StateType{};
    auto error_target = StateType{};
    derived()->save_state(t, x);
    while (t < tf) {
      // evaluate stages
      ode(x0, ks[0]);
      for (auto stage = 1; stage < n_stages; ++stage) {
        fill(temp_state, 0.0);
        for (auto j = 0; j < stage; ++j) {
          elementwise_mult_add(a[stage - 1][j], ks[j], temp_state);
        }
        scalar_mult(temp_state, dt);
        vector_add(x0, temp_state);
        ode(temp_state, ks[stage]);
      }

      // advance the state and compute the error estimate
      fill(x, 0.0);
      fill(error_estimate, 0.0);
      for (auto j = 0; j < n_stages; ++j) {
        elementwise_mult_add(b[j], ks[j], x);
        elementwise_mult_add(db()[j], ks[j], error_estimate);
      }
      scalar_mult(x, dt);
      vector_add(x0, x);
      scalar_mult(error_estimate, dt);

      // estimate error
      elementwise_binary_op(x0, x, error_target, [](auto a, auto b) {
        return std::max(std::abs(a), std::abs(b));
      });
      elementwise_mult(rtol, error_target);
      vector_add(atol, error_target);
      auto scaled_error = rk_norm(error_estimate, error_target);

      // accept or reject the step
      if (scaled_error <= 1.0) {
        t += dt;
        x0 = x;
        derived()->save_state(t, x);
      }

      // update step size
      auto dtnew =
          dt * safety_factor() / std::pow(scaled_error, 1.0 / (1.0 + q()));
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

  auto estimate_initial_step(StateType x0, StateType atol,
                             StateType rtol) -> double {
    // algorithm presented in Hairer II.4
    auto error_target = x0;
    elementwise_unary_op(error_target, [](auto& x) { x = std::abs(x); });
    elementwise_mult(rtol, error_target);
    vector_add(atol, error_target);

    auto f0 = StateType{};
    derived()->ode(x0, f0);
    auto d0 = rk_norm(x0, error_target);
    auto d1 = rk_norm(f0, error_target);
    auto dt0 = (d0 < 1.0e-5 or d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (d0 / d1);

    auto x1 = StateType{};
    elementwise_mult_add(dt0, f0, x0, x1);
    auto f1 = StateType{};
    derived()->ode(x1, f1);
    auto df = f1;
    elementwise_binary_op(f1, f0, df, [](auto a, auto b) { return a - b; });
    auto d2 = rk_norm(df, error_target) / dt0;

    auto constexpr p = RKMethod::p;
    auto dt1 = (std::max(d1, d2) <= 1.0e-15)
                   ? std::max(1.0e-6, dt0 * 1.0e-3)
                   : std::pow(0.01 / std::max(d1, d2), (1.0 / (1.0 + p)));
    return std::min(100.0 * dt0, dt1);
  }

  auto rk_norm(StateType v, StateType scale) -> double {
    elementwise_binary_op(scale, v, [](auto a, auto& b) { b /= a; });
    return std::sqrt(inner_product(v, v) / v.size());
  }

 protected:
  ~RKEmbedded() = default;

 private:
  auto static constexpr max_step_scale = 6.0;
  auto static constexpr min_step_scale = 0.33;

  auto static constexpr db() {
    auto static constexpr db = []() {
      auto db = RKMethod::b;
      for (auto&& [x, xt] : vws::zip(db, RKMethod::bt)) {
        x -= xt;
      }
      return db;
    }();
    return db;
  }

  auto static constexpr q() {
    auto static constexpr q = std::min(RKMethod::p, RKMethod::pt);
    return q;
  }

  auto static safety_factor() {
    auto static const safety_factor = std::pow(0.38, (1.0 / (1.0 + q())));
    return safety_factor;
  }

  auto* derived() { return static_cast<RKMethod*>(this); }
  auto* derived() const { return static_cast<RKMethod const*>(this); }
};
