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
    auto constexpr q = std::min(ButcherTableau::p, ButcherTableau::pt);
    auto static const safety_factor = std::pow(0.38, (1.0 / (1.0 + q)));
    auto static constexpr a = ButcherTableau::a;
    auto static constexpr b = ButcherTableau::b;
    auto static constexpr n_stages = ButcherTableau::n_stages;
    auto ks = StateContainer<typename OwnedState::StateType, n_stages>{};

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
      for (auto stage = 1; stage < n_stages; ++stage) {
        exe.call_parallel_kernel(
            [temp_state = temp_state.data(), ks = ks.data(), x0 = x0.data(),
             stage, dt](int i) {
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
          [x = x.data(), error_estimate = error_estimate.data(), ks = ks.data(),
           x0 = x0.data(), dt](int i) {
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
          [error_target = error_target.data(), atol = atol.data(),
           rtol = rtol.data(), x = x.data(), x0 = x0.data()](int i) {
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

  auto static rk_norm(ParallelExecutor& exe, OwnedState const& v,
                      OwnedState const& scale) {
    auto n_var = std::ssize(v);
    return std::sqrt(
        exe.transform_reduce(
            0.0, std::plus<>{},
            [v = v.data(), scale = scale.data()] ENABLE_CUDA(int i) {
              auto scaled_value = v[i] / scale[i];
              return scaled_value * scaled_value;
            },
            n_var) /
        n_var);
  }

  double static estimate_initial_step(ParallelExecutor& exe,
                                      OwnedState const& x0,
                                      OwnedState const& atol,
                                      OwnedState const& rtol, ODE& ode) {
    auto const n_var = std::ssize(x0);

    auto error_target = OwnedState{};
    exe.call_parallel_kernel(
        [error_target = error_target.data(), atol = atol.data(),
         rtol = rtol.data(), x0 = x0.data()] ENABLE_CUDA(int i) {
          error_target[i] = atol[i] + rtol[i] * std::abs(x0[i]);
        },
        n_var);

    auto f0 = OwnedState{};
    ode(x0, f0);
    auto d0 = rk_norm(exe, x0, error_target);
    auto d1 = rk_norm(exe, f0, error_target);
    auto dt0 = (d0 < 1.0e-5 or d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (d0 / d1);

    auto x1 = OwnedState{};
    exe.call_parallel_kernel([x1 = x1.data(), x0 = x0.data(), f0 = f0.data(),
                              dt0](int i) { x1[i] = x0[i] + f0[i] * dt0; },
                             n_var);
    auto df = OwnedState{};
    ode(x1, df);
    exe.call_parallel_kernel(
        [df = df.data(), f0 = f0.data()](int i) { df[i] -= f0[i]; }, n_var);
    auto d2 = rk_norm(exe, df, error_target) / dt0;

    auto constexpr p = ButcherTableau::p;
    auto dt1 = (std::max(d1, d2) <= 1.0e-15)
                   ? std::max(1.0e-6, dt0 * 1.0e-3)
                   : std::pow(0.01 / std::max(d1, d2), (1.0 / (1.0 + p)));
    return std::min(100.0 * dt0, dt1);
  }
};
