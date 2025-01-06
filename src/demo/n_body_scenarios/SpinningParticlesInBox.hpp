#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <string>

template <int N, template <typename, int> typename StateContainer,
          typename ValueType, ValueType sof_divisor, ValueType tol>
struct SpinningParticlesInBox {
  static inline auto const name = std::string{"SpinningParticlesInBox"};
  static constexpr auto L = 1.0;
  static constexpr auto n_particles = N;
  static constexpr auto n_var = N * 6;
  static inline auto const tf = 10.0 * std::sqrt(L * L * L / N);
  static inline auto const omega = 2.0 / std::sqrt(L * L * L / N);
  static constexpr auto softening = L / (N * (N - 1)) / sof_divisor;
  static constexpr auto tolerance_value = tol;

  StateContainer<ValueType, n_var> initial_state;
  StateContainer<ValueType, n_var> tolerance_array;

  SpinningParticlesInBox() {
    auto gen = std::mt19937{0};
    auto dist = std::uniform_real_distribution<ValueType>(0.0, L);

    // Current interface requires placing random values in std::array before
    // copying to StateContainer. std::unique_ptr is used to prevent stack
    // overflow for large arrays. Note: cuRAND should not be used if it produces
    // different values from the host library.
    // TODO: make this simpler.
    auto init_array_ptr = std::make_unique<std::array<ValueType, n_var>>();
    // initialize positions
    for (auto i = 0; i < n_var / 2; ++i) {
      (*init_array_ptr)[i] = dist(gen);
    }
    // add spin about vertical axis
    for (auto i = 0; i < n_particles; ++i) {
      auto const x = (*init_array_ptr)[3 * i] - L / 2.0;
      auto const y = (*init_array_ptr)[3 * i + 1] - L / 2.0;
      (*init_array_ptr)[n_var / 2 + 3 * i] = -omega * y;
      (*init_array_ptr)[n_var / 2 + 3 * i + 1] = omega * x;
      (*init_array_ptr)[n_var / 2 + 3 * i + 2] = 0.0;
    }
    initial_state = StateContainer<ValueType, n_var>{*init_array_ptr};

    auto tol_array_ptr = std::make_unique<std::array<ValueType, n_var>>();
    std::fill((*tol_array_ptr).begin(), (*tol_array_ptr).end(),
              tolerance_value);
    tolerance_array = StateContainer<ValueType, n_var>{*tol_array_ptr};
  }
};
