#pragma once

#include <cmath>
#include <span>

#include "ParallelExecutor.hpp"

template <typename ValueType, int n_var>
struct NBodySimpleODE {
  double softening{};

  constexpr void operator()(auto& exe, std::span<ValueType const, n_var> x,
                            std::span<ValueType, n_var> dxdt) {
    constexpr auto vel_offset = n_var / 2;
    call_parallel_kernel<init_dxdt_kernel>(exe, vel_offset, vel_offset, x,
                                           dxdt);

    constexpr auto n_particles = n_var / 6;
    call_parallel_kernel<acc_kernel>(exe, n_particles, n_particles, x, dxdt,
                                     softening * softening);
  }

  static constexpr void init_dxdt_kernel(int i, int vel_offset,
                                         std::span<ValueType const, n_var> x,
                                         std::span<ValueType, n_var> dxdt) {
    dxdt[i] = x[i + vel_offset];
    dxdt[i + vel_offset] = 0.0;
  }

  static constexpr void acc_kernel(int i, int n_particles,
                                   std::span<ValueType const, n_var> x,
                                   std::span<ValueType, n_var> dxdt,
                                   double softening_sq) {
    auto xi = x[3 * i];
    auto yi = x[3 * i + 1];
    auto zi = x[3 * i + 2];

    for (auto j = 0; j < n_particles; ++j) {
      if (i == j) {
        continue;
      }
      auto xj = x[3 * j];
      auto yj = x[3 * j + 1];
      auto zj = x[3 * j + 2];

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

      auto a_offset = 3 * n_particles;
      dxdt[a_offset + 3 * i] += ax;
      dxdt[a_offset + 3 * i + 1] += ay;
      dxdt[a_offset + 3 * i + 2] += az;
    }
  }
};
