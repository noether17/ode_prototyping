#pragma once

#include <cmath>
#include <span>

#include "AtomicUtil.hpp"
#include "ParallelExecutor.hpp"

template <typename ValueType, std::size_t n_var
          /*auto masses*/>
struct NBodyODE {
  static constexpr auto n_dof = n_var / 2;
  static constexpr auto n_particles = n_var / 6;
  static constexpr auto n_pairs = n_particles * (n_particles - 1) / 2;

  using state_span = std::span<ValueType, n_var>;
  using const_state_span = std::span<ValueType const, n_var>;
  using vector_span = std::span<ValueType, n_dof>;
  using const_vector_span = std::span<ValueType const, n_dof>;

  /* Whenever arbitrarily close approaches are possible, a softening parameter
   * is required to prevent forces from becoming too large so that the step size
   * remains reasonable. The softening parameter represents the length scale at
   * which particles appear as diffuse objects. To minimize the physical impact
   * of the softening parameter, choose it to be roughly the separation between
   * two particles in orbit around each other with all other particles scattered
   * to infinity with zero energy. This is a good approximation of the minimum
   * distance of physical interest in the simulation. */
  double softening{};

  constexpr void operator()(auto& exe, const_state_span x,
                            state_span dxdt) const {
    call_parallel_kernel<detail::nbody_init_dxdt_kernel>(exe, n_dof, x, dxdt);

    auto pos = x.template subspan<0, n_dof>();
    auto acc = dxdt.template subspan<n_dof, n_dof>();
    call_parallel_kernel<detail::nbody_acc_kernel>(
        exe, n_pairs, n_particles, pos, acc, softening * softening);
  }

  constexpr void acc(auto& exe, const_state_span pos_vel, vector_span acc) {
    call_parallel_kernel<detail::nbody_zero_vector_kernel>(exe, n_dof, acc);
    auto pos = pos_vel.template subspan<0, n_dof>();
    call_parallel_kernel<detail::nbody_acc_kernel>(
        exe, n_pairs, n_particles, pos, acc, softening * softening);
  }

  struct detail {
    /* Initializes the velocity portion of dxdt to the velocity portion of the
     * state, x, and initializes the acceleration portion of dxdt to zero. */
    static constexpr void nbody_init_dxdt_kernel(std::size_t i,
                                                 const_state_span x,
                                                 state_span dxdt) {
      dxdt[i] = x[i + n_dof];
      dxdt[i + n_dof] = 0.0;
    }

    /* Used to initialize acceleration vector to zero when computing
     * acceleration independently. */
    static constexpr void nbody_zero_vector_kernel(std::size_t i,
                                                   vector_span v) {
      v[i] = 0.0;
    }

    /* Computes the acceleration values for a single pair of particles and
     * atomically adds the contribution of the pair to the acceleration portion
     * of dxdt. */
    static constexpr void nbody_acc_kernel(std::size_t pair_id,
                                           std::size_t n_particles,
                                           const_vector_span pos,
                                           vector_span acc,
                                           double softening_sq) {
      // compute indices
      auto n_minus_half = n_particles - 0.5;
      auto i = static_cast<std::size_t>(
          n_minus_half -
          std::sqrt(n_minus_half * n_minus_half - 2.0 * pair_id));
      auto j = pair_id - (n_particles - 1) * i + (i * (i + 1)) / 2 + 1;

      // read coordinates
      auto xi = pos[3 * i];
      auto yi = pos[3 * i + 1];
      auto zi = pos[3 * i + 2];
      auto xj = pos[3 * j];
      auto yj = pos[3 * j + 1];
      auto zj = pos[3 * j + 2];

      // compute acceleration per mass
      auto dx = xj - xi;
      auto dy = yj - yi;
      auto dz = zj - zi;
      auto dist_sq =
          dx * dx + dy * dy + dz * dz + softening_sq;  // Plummer sphere model.
      auto dist = std::sqrt(dist_sq);
      auto denominator = dist * dist_sq;
      auto ax = dx / denominator;
      auto ay = dy / denominator;
      auto az = dz / denominator;

      // compute acceleration values
      auto axi = ax;
      auto ayi = ay;
      auto azi = az;
      // if constexpr (masses.size() == 1) {
      //   axi *= masses[0];
      //   ayi *= masses[0];
      //   azi *= masses[0];
      // } else if constexpr (masses.size() > 1) {
      //   axi *= masses[j];
      //   ayi *= masses[j];
      //   azi *= masses[j];
      // }
      auto axj = -ax;
      auto ayj = -ay;
      auto azj = -az;
      // if constexpr (masses.size() == 1) {
      //   axj *= masses[0];
      //   ayj *= masses[0];
      //   azj *= masses[0];
      // } else if constexpr (masses.size() > 1) {
      //   axj *= masses[i];
      //   ayj *= masses[i];
      //   azj *= masses[i];
      // }

      // add contribution of pair to acceleration
      au::atomic_add(&acc[3 * i], axi);
      au::atomic_add(&acc[3 * i + 1], ayi);
      au::atomic_add(&acc[3 * i + 2], azi);
      au::atomic_add(&acc[3 * j], axj);
      au::atomic_add(&acc[3 * j + 1], ayj);
      au::atomic_add(&acc[3 * j + 2], azj);
    }
  };
};
