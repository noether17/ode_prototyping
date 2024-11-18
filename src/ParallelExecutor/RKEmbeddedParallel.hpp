#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <vector>

#ifdef __CUDA_ARCH__
#define ENABLE_CUDA __host__ __device__
#else
#define ENABLE_CUDA
#endif

template <template <typename, int> typename StateContainer, typename ValueType,
          int NVAR, typename ButcherTableau, typename ODE, typename Output,
          typename ParallelExecutor>
class RKEmbeddedParallel {
 public:
  using OwnedState = StateContainer<ValueType, NVAR>;
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
    auto ks = StateContainer<typename OwnedState::StateType,
                             ButcherTableau::n_stages>{};

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
      ode(x0, ks[0]);
      for (auto stage = 1; stage < ButcherTableau::n_stages; ++stage) {
        exe.call_parallel_kernel(
            [stage, dt] ENABLE_CUDA(
                int i, std::span<ValueType, NVAR> temp_state,
                std::span<typename decltype(ButcherTableau::a)::value_type,
                          std::ssize(ButcherTableau::a)>
                    a,
                std::span<typename OwnedState::StateType,
                          ButcherTableau::n_stages>
                    ks,
                std::span<ValueType, NVAR> x0) {
              temp_state[i] = 0.0;
              for (auto j = 0; j < stage; ++j) {
                temp_state[i] += a[stage - 1][j] * ks[j][i];
              }
              temp_state[i] = x0[i] + temp_state[i] * dt;
            },
            n_var, temp_state, state_a, ks, x0);
        ode(temp_state, ks[stage]);
      }

      // advance the state and compute the error estimate
      exe.call_parallel_kernel(
          [dt] ENABLE_CUDA(
              int i, std::span<ValueType, NVAR> x,
              std::span<ValueType, NVAR> error_estimate,
              std::span<typename decltype(ButcherTableau::b)::value_type const,
                        std::ssize(ButcherTableau::b)>
                  b,
              std::span<typename decltype(db)::value_type const, std::ssize(db)>
                  db,
              std::span<typename OwnedState::StateType const,
                        ButcherTableau::n_stages>
                  ks,
              std::span<ValueType const, NVAR> x0) {
            x[i] = 0.0;
            error_estimate[i] = 0.0;
            for (auto j = 0; j < ButcherTableau::n_stages; ++j) {
              x[i] += b[j] * ks[j][i];
              error_estimate[i] += db[j] * ks[j][i];
            }
            x[i] = x0[i] + x[i] * dt;
            error_estimate[i] *= dt;
          },
          n_var, x, error_estimate, state_b, state_db, ks, x0);

      // estimate error
      exe.call_parallel_kernel(
          [] ENABLE_CUDA(int i, std::span<ValueType, NVAR> error_target,
                         std::span<ValueType const, NVAR> atol,
                         std::span<ValueType const, NVAR> rtol,
                         std::span<ValueType const, NVAR> x,
                         std::span<ValueType const, NVAR> x0) {
            error_target[i] =
                atol[i] + rtol[i] * std::max(std::abs(x[i]), std::abs(x0[i]));
          },
          n_var, error_target, atol, rtol, x, x0);
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

  ValueType static rk_norm(ParallelExecutor& exe,
                           std::span<ValueType const, NVAR> v,
                           std::span<ValueType const, NVAR> scale) {
    auto n_var = std::ssize(v);
    return std::sqrt(exe.transform_reduce(
                         0.0, std::plus<>{},
                         [v, scale] ENABLE_CUDA(int i) {
                           auto scaled_value = v[i] / scale[i];
                           return scaled_value * scaled_value;
                         },
                         n_var) /
                     n_var);
  }

  double static estimate_initial_step(ParallelExecutor& exe,
                                      std::span<ValueType const, NVAR> x0,
                                      std::span<ValueType const, NVAR> atol,
                                      std::span<ValueType const, NVAR> rtol,
                                      ODE& ode) {
    auto const n_var = std::ssize(x0);

    auto error_target = OwnedState{};
    exe.call_parallel_kernel(
        [atol, rtol, x0] ENABLE_CUDA(int i,
                                     std::span<ValueType, NVAR> error_target) {
          error_target[i] = atol[i] + rtol[i] * std::abs(x0[i]);
        },
        n_var, error_target);

    auto f0 = OwnedState{};
    ode(x0, f0);
    auto d0 = rk_norm(exe, x0, error_target);
    auto d1 = rk_norm(exe, f0, error_target);
    auto dt0 = (d0 < 1.0e-5 or d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (d0 / d1);

    auto x1 = OwnedState{};
    exe.call_parallel_kernel(
        [x0, dt0] ENABLE_CUDA(int i, std::span<ValueType, NVAR> x1,
                              std::span<ValueType const, NVAR> f0) {
          x1[i] = x0[i] + f0[i] * dt0;
        },
        n_var, x1, f0);
    auto df = OwnedState{};
    ode(x1, df);
    exe.call_parallel_kernel(
        [] ENABLE_CUDA(int i, std::span<ValueType, NVAR> df,
                       std::span<ValueType const, NVAR> f0) { df[i] -= f0[i]; },
        n_var, df, f0);
    auto d2 = rk_norm(exe, df, error_target) / dt0;

    auto constexpr p = ButcherTableau::p;
    auto dt1 = (std::max(d1, d2) <= 1.0e-15)
                   ? std::max(1.0e-6, dt0 * 1.0e-3)
                   : std::pow(0.01 / std::max(d1, d2), (1.0 / (1.0 + p)));
    return std::min(100.0 * dt0, dt1);
  }
};
