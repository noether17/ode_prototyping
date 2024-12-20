#pragma once

#include <array>
#include <cmath>
#include <span>
#include <vector>

template <template <typename, int> typename StateContainer, typename ValueType,
          int NVAR, typename ButcherTableau, typename ODE, typename Output,
          typename ParallelExecutor>
class RKEmbeddedParallel {
 public:
  using OwnedState = StateContainer<ValueType, NVAR>;
  //[stage, dt](
  //    int i, std::span<ValueType, NVAR> temp_state,
  //    std::span<typename decltype(ButcherTableau::a)::value_type,
  //              std::ssize(ButcherTableau::a)>
  //        a,
  //    std::span<typename OwnedState::StateType,
  //              ButcherTableau::n_stages>
  //        ks,
  //    std::span<ValueType, NVAR> x0) {
  //  temp_state[i] = 0.0;
  //  for (auto j = 0; j < stage; ++j) {
  //    temp_state[i] += a[stage - 1][j] * ks[j][i];
  //  }
  //  temp_state[i] = x0[i] + temp_state[i] * dt;
  //},

  auto static constexpr rk_stage_kernel(
      int i, int stage, ValueType dt, ValueType* temp_state,
      typename decltype(ButcherTableau::a)::value_type const* a, ValueType* ks,
      ValueType const* x0) {
    temp_state[i] = 0.0;
    for (auto j = 0; j < stage; ++j) {
      temp_state[i] += a[stage - 1][j] * ks[j * NVAR + i];
    }
    temp_state[i] = x0[i] + temp_state[i] * dt;
  }

  //[dt](
  //    int i, std::span<ValueType, NVAR> x,
  //    std::span<ValueType, NVAR> error_estimate,
  //    std::span<typename decltype(ButcherTableau::b)::value_type const,
  //              std::ssize(ButcherTableau::b)>
  //        b,
  //    std::span<typename decltype(db)::value_type const, std::ssize(db)>
  //        db,
  //    std::span<typename OwnedState::StateType const,
  //              ButcherTableau::n_stages>
  //        ks,
  //    std::span<ValueType const, NVAR> x0) {
  //  x[i] = 0.0;
  //  error_estimate[i] = 0.0;
  //  for (auto j = 0; j < ButcherTableau::n_stages; ++j) {
  //    x[i] += b[j] * ks[j][i];
  //    error_estimate[i] += db[j] * ks[j][i];
  //  }
  //  x[i] = x0[i] + x[i] * dt;
  //  error_estimate[i] *= dt;
  //},
  auto static constexpr update_state_and_error_kernel(
      int i, ValueType dt, ValueType* x, ValueType* error_estimate,
      typename decltype(ButcherTableau::b)::value_type const* b,
      typename decltype(ButcherTableau::b)::value_type const* db, ValueType* ks,
      ValueType const* x0) {
    x[i] = 0.0;
    error_estimate[i] = 0.0;
    for (auto j = 0; j < ButcherTableau::n_stages; ++j) {
      x[i] += b[j] * ks[j * NVAR + i];
      error_estimate[i] += db[j] * ks[j * NVAR + i];
    }
    x[i] = x0[i] + x[i] * dt;
    error_estimate[i] *= dt;
  }

  //[](int i, std::span<ValueType, NVAR> error_target,
  //   std::span<ValueType const, NVAR> atol,
  //   std::span<ValueType const, NVAR> rtol,
  //   std::span<ValueType const, NVAR> x,
  //   std::span<ValueType const, NVAR> x0) {
  //  error_target[i] =
  //      atol[i] + rtol[i] * std::max(std::abs(x[i]), std::abs(x0[i]));
  //},
  auto static constexpr update_error_target_kernel(
      int i, ValueType* error_target, ValueType const* atol,
      ValueType const* rtol, ValueType const* x, ValueType const* x0) {
    error_target[i] =
        atol[i] + rtol[i] * std::max(std::abs(x[i]), std::abs(x0[i]));
  }

  void integrate(OwnedState x0, double t0, double tf, OwnedState atol,
                 OwnedState rtol, ODE ode, Output& output,
                 ParallelExecutor& exe) {
    auto static constexpr max_step_scale = 6.0;
    auto static constexpr min_step_scale = 0.33;
    auto static constexpr db = [] {
      auto db = ButcherTableau::b;
      for (auto i = 0; i < ButcherTableau::n_stages; ++i) {
        db[i] -= ButcherTableau::bt[i];
      }
      return db;
    }();
    auto state_a =
        StateContainer<typename decltype(ButcherTableau::a)::value_type,
                       std::ssize(ButcherTableau::a)>{ButcherTableau::a};
    auto state_b =
        StateContainer<typename decltype(ButcherTableau::b)::value_type,
                       std::ssize(ButcherTableau::b)>{ButcherTableau::b};
    auto state_db =
        StateContainer<typename decltype(db)::value_type, std::ssize(db)>{db};
    auto constexpr q = std::min(ButcherTableau::p, ButcherTableau::pt);
    auto static const safety_factor = std::pow(0.38, (1.0 / (1.0 + q)));
    auto ks = StateContainer<ValueType, ButcherTableau::n_stages * NVAR>{};

    auto dt = estimate_initial_step(exe, x0, atol, rtol, ode);
    auto t = t0;
    auto x = x0;
    auto n_var = std::ssize(x);
    auto temp_state = OwnedState{};
    auto error_estimate = OwnedState{};
    auto error_target = OwnedState{};
    output.save_state(t, x);
    while (t < tf) {
      // evaluate stages
      ode(exe, x0, ks.data());
      for (auto stage = 1; stage < ButcherTableau::n_stages; ++stage) {
        exe.template call_parallel_kernel<rk_stage_kernel>(
            n_var, stage, dt, temp_state.data(), state_a.data(), ks.data(),
            x0.data());
        ode(exe, temp_state, ks.data() + stage * NVAR);
      }

      // advance the state and compute the error estimate
      exe.template call_parallel_kernel<update_state_and_error_kernel>(
          n_var, dt, x.data(), error_estimate.data(), state_b.data(),
          state_db.data(), ks.data(), x0.data());

      // estimate error
      exe.template call_parallel_kernel<update_error_target_kernel>(
          n_var, error_target.data(), atol.data(), rtol.data(), x.data(),
          x0.data());
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

  auto static constexpr scaled_value_squared_kernel(int i, ValueType const* v,
                                                    ValueType const* scale) {
    auto scaled_value = v[i] / scale[i];
    return scaled_value * scaled_value;
  }

  auto static constexpr add(ValueType a, ValueType b) { return a + b; }

  ValueType static rk_norm(ParallelExecutor& exe,
                           std::span<ValueType const, NVAR> v,
                           std::span<ValueType const, NVAR> scale) {
    auto n_var = std::ssize(v);
    return std::sqrt(exe.template transform_reduce<ValueType, add,
                                                   scaled_value_squared_kernel>(
                         0.0, n_var, v.data(), scale.data()) /
                     n_var);
  }

  auto static constexpr compute_error_target_kernel(int i,
                                                    ValueType* error_target,
                                                    ValueType const* x0,
                                                    ValueType const* atol,
                                                    ValueType const* rtol) {
    error_target[i] = std::abs(x0[i]) * rtol[i] + atol[i];
  }

  auto static constexpr euler_step_kernel(int i, ValueType* x1,
                                          ValueType const* x0,
                                          ValueType const* f0, ValueType dt0) {
    x1[i] = x0[i] + f0[i] * dt0;
  }

  auto static constexpr difference_kernel(int i, ValueType* df,
                                          ValueType const* f0) {
    df[i] -= f0[i];
  }

  double static estimate_initial_step(ParallelExecutor& exe,
                                      std::span<ValueType const, NVAR> x0,
                                      std::span<ValueType const, NVAR> atol,
                                      std::span<ValueType const, NVAR> rtol,
                                      ODE& ode) {
    auto const n_var = std::ssize(x0);

    auto error_target = OwnedState{};
    exe.template call_parallel_kernel<compute_error_target_kernel>(
        n_var, error_target.data(), x0.data(), atol.data(), rtol.data());

    auto f0 = OwnedState{};
    ode(exe, x0, f0.data());
    auto d0 = rk_norm(exe, x0, error_target);
    auto d1 = rk_norm(exe, f0, error_target);
    auto dt0 = (d0 < 1.0e-5 or d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (d0 / d1);

    auto x1 = OwnedState{};
    exe.template call_parallel_kernel<euler_step_kernel>(
        n_var, x1.data(), x0.data(), f0.data(), dt0);
    auto df = OwnedState{};
    ode(exe, x1, df.data());
    exe.template call_parallel_kernel<difference_kernel>(n_var, df.data(),
                                                         f0.data());
    auto d2 = rk_norm(exe, df, error_target) / dt0;

    auto constexpr p = ButcherTableau::p;
    auto dt1 = (std::max(d1, d2) <= 1.0e-15)
                   ? std::max(1.0e-6, dt0 * 1.0e-3)
                   : std::pow(0.01 / std::max(d1, d2), (1.0 / (1.0 + p)));
    return std::min(100.0 * dt0, dt1);
  }
};
