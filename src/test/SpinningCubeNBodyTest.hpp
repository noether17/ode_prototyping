#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <random>
#include <span>

#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "RawOutput.hpp"

// Defines parameters for NBody test.
template <int N, template <typename, std::size_t> typename StateType>
struct SpinningCubeNBodyTest {
  static constexpr auto n_particles = N;
  static constexpr auto n_var = N * 6;
  static constexpr auto L = 1.0;

  static constexpr auto t0 = 0.0;
  // sqrt(L^3 / N) is a rough estimate of the dynamical time scale from
  // dimensional analysis. Let the simulation go an order of magnitude longer
  // than this to see interesting evolution.
  static inline auto const tf = std::sqrt(L * L * L / N);

  // Rough estimate of the angular velocity required to keep a
  // spherically-symmetrical uniform mass distribution in equilibrium (not meant
  // to be exact, just meant to prevent rapid collapse of the initial mass
  // distribution).
  static inline auto const omega =
      2.0 * std::sqrt(std::numbers::pi / 3.0) / std::sqrt(L * L * L / N);

  // Softening parameter from Power, et al.
  static inline auto const softening = 4.0 * L / std::sqrt(n_particles);

  static inline auto const x0 = [] {
    auto gen = std::mt19937{0};
    auto dist = std::uniform_real_distribution<double>(0.0, L);

    // initialize positions
    auto x0_data_ptr = std::make_unique<std::array<double, n_var>>();
    for (auto i = 0; i < n_var / 2; ++i) {
      (*x0_data_ptr)[i] = dist(gen);
    }

    // add spin about vertical axis
    for (auto i = 0; i < n_particles; ++i) {
      auto const x = (*x0_data_ptr)[3 * i] - L / 2.0;
      auto const y = (*x0_data_ptr)[3 * i + 1] - L / 2.0;
      (*x0_data_ptr)[n_var / 2 + 3 * i] = -omega * y;
      (*x0_data_ptr)[n_var / 2 + 3 * i + 1] = omega * x;
      (*x0_data_ptr)[n_var / 2 + 3 * i + 2] = 0.0;
    }

    return StateType{*x0_data_ptr};
  }();

  // tolerance
  static inline auto const tol = softening;
  static inline auto const atol = [] {
    auto tol_array_ptr = std::make_unique<std::array<double, n_var>>();
    std::ranges::fill(*tol_array_ptr, softening);
    return StateType{*tol_array_ptr};
  }();
  static inline auto const rtol = atol;

  static inline auto const softened_nbody_ode =
      NBodyODE<double, n_var>{softening};
  auto operator()(auto& exe, std::span<double const, n_var> x,
                  std::span<double, n_var> dxdt) {
    softened_nbody_ode(exe, x, dxdt);
  }

  RawOutput<HeapState<double, n_var>> output{};
};
