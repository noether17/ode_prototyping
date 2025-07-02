#pragma once

#include <utility>

#include "ODEState.hpp"
#include "ParallelExecutor.hpp"

struct ParallelLeapfrogIntegrator {
  template <ODEState StateType, typename ODE, typename Output,
            typename ParallelExecutor>
  static void integrate(StateType x0,
                        ode_state_traits<StateType>::value_type t0,
                        ode_state_traits<StateType>::value_type tf,
                        ode_state_traits<StateType>::value_type dt, ODE ode,
                        Output& output, ParallelExecutor& exe) {
    static constexpr auto n_var = x0.size();
    static constexpr auto n_dof = n_var / 2;
    auto t = t0;
    auto& x = x0;
    output.save_state(t, x);
    auto const pos_span = span(x).template subspan<0, n_dof>();
    auto const vel_span = span(x).template subspan<n_dof, n_dof>();

    auto accelerations = StateType{};
    auto const prev_acc_span = span(accelerations).template subspan<0, n_dof>();
    auto const curr_acc_span =
        span(accelerations).template subspan<n_dof, n_dof>();
    ode.acc(exe, span(x), prev_acc_span);
    while (t < tf) {
      call_parallel_kernel<detail<StateType>::update_position_kernel>(
          exe, n_dof, dt, pos_span, vel_span, prev_acc_span);
      ode.acc(exe, span(x), curr_acc_span);
      call_parallel_kernel<detail<StateType>::update_velocity_kernel>(
          exe, n_dof, dt, vel_span, prev_acc_span, curr_acc_span);
      call_parallel_kernel<detail<StateType>::update_acceleration_kernel>(
          exe, n_dof, prev_acc_span, curr_acc_span);

      t += dt;
      output.save_state(t, x);
    }
  }

  template <ODEState StateType>
  struct detail {
    using ODEStateTraits = ode_state_traits<StateType>;
    using ValueType = ODEStateTraits::value_type;
    static constexpr auto n_var = ODEStateTraits::size;
    static constexpr auto n_dof = n_var / 2;
    using StateSpan = ODEStateTraits::span_type;
    using ConstStateSpan = ODEStateTraits::const_span_type;

    using DofSpan =
        decltype(std::declval<StateSpan>().template subspan<0, n_dof>());
    using ConstDofSpan =
        decltype(std::declval<ConstStateSpan>().template subspan<0, n_dof>());

    static constexpr void update_position_kernel(std::size_t i, ValueType dt,
                                                 DofSpan pos, ConstDofSpan vel,
                                                 ConstDofSpan acc) {
      pos[i] += dt * (vel[i] + 0.5 * dt * acc[i]);
    }

    static constexpr void update_velocity_kernel(std::size_t i, ValueType dt,
                                                 DofSpan vel, ConstDofSpan acc0,
                                                 ConstDofSpan acc1) {
      vel[i] += 0.5 * dt * (acc0[i] + acc1[i]);
    }

    static constexpr void update_acceleration_kernel(std::size_t i,
                                                     DofSpan acc0,
                                                     ConstDofSpan acc1) {
      acc0[i] = acc1[i];
    }
  };
};
