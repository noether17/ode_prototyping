#pragma once

#include <array>
#include <cmath>
#include <span>

#include "AtomicUtil.hpp"
#include "ParallelExecutor.hpp"

template <typename ValueType, int n_var,
          /*auto masses,*/ double softening = 0.0>
struct NBodyODE {
  constexpr void operator()(auto& exe, std::span<ValueType const, n_var> x,
                            std::span<ValueType, n_var> dxdt) {
    constexpr auto vel_offset = n_var / 2;
    call_parallel_kernel<nbody_init_dxdt_kernel>(exe, vel_offset, vel_offset, x,
                                                 dxdt);

    constexpr auto n_particles = n_var / 6;
    constexpr auto n_pairs = n_particles * (n_particles - 1) / 2;
    call_parallel_kernel<nbody_acc_kernel>(exe, n_pairs, n_particles, x, dxdt);
  }

  /* Whenever arbitrarily close approaches are possible, a softening parameter
   * is required to prevent forces from becoming too large so that the step size
   * remains reasonable. The softening parameter represents the length scale at
   * which particles appear as diffuse objects. To minimize the physical impact
   * of the softening parameter, choose it to be roughly the separation between
   * two particles in orbit around each other with all other particles scattered
   * to infinity with zero energy. This is a good approximation of the minimum
   * distance of physical interest in the simulation. */
  static constexpr auto softening_sq = softening * softening;

  /* Initializes the velocity portion of dxdt to the velocity portion of the
   * state, x, and initializes the acceleration portion of dxdt to zero. */
  static constexpr void nbody_init_dxdt_kernel(
      int i, int vel_offset, std::span<ValueType const, n_var> x,
      std::span<ValueType, n_var> dxdt) {
    dxdt[i] = x[i + vel_offset];
    dxdt[i + vel_offset] = 0.0;
  }

  /* Computes the acceleration values for a single pair of particles and
   * atomically adds the contribution of the pair to the acceleration portion of
   * dxdt. */
  static constexpr void nbody_acc_kernel(int pair_id, int n_particles,
                                         std::span<ValueType const, n_var> x,
                                         std::span<ValueType, n_var> dxdt) {
    // compute indices
    auto n_minus_half = n_particles - 0.5;
    auto i = static_cast<int>(
        n_minus_half - std::sqrt(n_minus_half * n_minus_half - 2.0 * pair_id));
    auto j = pair_id - (n_particles - 1) * i + (i * (i + 1)) / 2 + 1;

    // read coordinates
    auto xi = x[3 * i];
    auto yi = x[3 * i + 1];
    auto zi = x[3 * i + 2];
    auto xj = x[3 * j];
    auto yj = x[3 * j + 1];
    auto zj = x[3 * j + 2];

    // compute acceleration per mass
    auto dx = xj - xi;
    auto dy = yj - yi;
    auto dz = zj - zi;
    auto dist_sq = dx * dx + dy * dy + dz * dz;
    auto dist = std::sqrt(dist_sq);
    auto denominator = dist * (dist_sq + softening_sq);
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
    auto a_offset = 3 * n_particles;
    au::atomic_add(&dxdt[a_offset + 3 * i], axi);
    au::atomic_add(&dxdt[a_offset + 3 * i + 1], ayi);
    au::atomic_add(&dxdt[a_offset + 3 * i + 2], azi);
    au::atomic_add(&dxdt[a_offset + 3 * j], axj);
    au::atomic_add(&dxdt[a_offset + 3 * j + 1], ayj);
    au::atomic_add(&dxdt[a_offset + 3 * j + 2], azj);
  }
};
