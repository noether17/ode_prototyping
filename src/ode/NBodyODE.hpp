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

  static constexpr auto softening_sq = softening * softening;

  static constexpr void nbody_init_dxdt_kernel(
      int i, int vel_offset, std::span<ValueType const, n_var> x,
      std::span<ValueType, n_var> dxdt) {
    dxdt[i] = x[i + vel_offset];
    dxdt[i + vel_offset] = 0.0;
  }

  static constexpr void nbody_acc_kernel(int pair_id, int n_particles,
                                         std::span<ValueType const, n_var> x,
                                         std::span<ValueType, n_var> dxdt) {
    // compute indices
    auto n_minus_half = n_particles - 0.5;
    auto i = static_cast<int>(
        n_minus_half - std::sqrt(n_minus_half * n_minus_half - 2.0 * pair_id));
    auto j = pair_id - (n_particles - 1) * i + (i * (i + 1)) / 2 + 1;

    // read coordinates
    auto ix = x[3 * i];
    auto iy = x[3 * i + 1];
    auto iz = x[3 * i + 2];
    auto jx = x[3 * j];
    auto jy = x[3 * j + 1];
    auto jz = x[3 * j + 2];

    // compute acceleration per mass
    auto dx = jx - ix;
    auto dy = jy - iy;
    auto dz = jz - iz;
    auto dist_sq = dx * dx + dy * dy + dz * dz;
    auto dist = std::sqrt(dist_sq);
    auto denominator = dist * (dist_sq + softening_sq);
    auto ax = dx / denominator;
    auto ay = dy / denominator;
    auto az = dz / denominator;

    // compute acceleration values
    auto iax = ax;
    auto iay = ay;
    auto iaz = az;
    // if constexpr (masses.size() == 1) {
    //   iax *= masses[0];
    //   iay *= masses[0];
    //   iaz *= masses[0];
    // } else if constexpr (masses.size() > 1) {
    //   iax *= masses[j];
    //   iay *= masses[j];
    //   iaz *= masses[j];
    // }
    auto jax = -ax;
    auto jay = -ay;
    auto jaz = -az;
    // if constexpr (masses.size() == 1) {
    //   jax *= masses[0];
    //   jay *= masses[0];
    //   jaz *= masses[0];
    // } else if constexpr (masses.size() > 1) {
    //   jax *= masses[i];
    //   jay *= masses[i];
    //   jaz *= masses[i];
    // }

    // add contribution of pair to acceleration
    auto a_offset = 3 * n_particles;
    au::atomic_add(&dxdt[a_offset + 3 * i], iax);
    au::atomic_add(&dxdt[a_offset + 3 * i + 1], iay);
    au::atomic_add(&dxdt[a_offset + 3 * i + 2], iaz);
    au::atomic_add(&dxdt[a_offset + 3 * j], jax);
    au::atomic_add(&dxdt[a_offset + 3 * j + 1], jay);
    au::atomic_add(&dxdt[a_offset + 3 * j + 2], jaz);
  }
};
