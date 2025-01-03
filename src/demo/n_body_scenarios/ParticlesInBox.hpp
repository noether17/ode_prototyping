#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <string>

template <int N, template <typename, int> typename StateContainer,
          typename ValueType>
struct ParticlesInBox {
  static inline auto const name = std::string{"ParticlesInBox"};
  static constexpr auto L = 1.0;
  static constexpr auto n_particles = N;
  static constexpr auto n_var = N * 6;
  static inline auto const tf = 10.0 * std::sqrt(L * L * L / N);
  static constexpr auto softening = L / (N * (N - 1));
  static constexpr auto tolerance_value = 1.0e-3;

  StateContainer<ValueType, n_var> initial_state;
  StateContainer<ValueType, n_var> tolerance_array;

  ParticlesInBox() {
    auto gen = std::mt19937{0};
    auto dist = std::uniform_real_distribution<ValueType>(0.0, L);

    // Current interface requires placing random values in std::array before
    // copying to StateContainer. std::unique_ptr is used to prevent stack
    // overflow for large arrays. Note: cuRAND should not be used if it produces
    // different values from the host library.
    // TODO: make this simpler.
    auto init_array_ptr = std::make_unique<std::array<ValueType, n_var>>();
    for (auto i = 0; i < n_var / 2; ++i) {
      (*init_array_ptr)[i] = dist(gen);
      (*init_array_ptr)[i + n_var / 2] = 0.0;
    }
    initial_state = StateContainer<ValueType, n_var>{*init_array_ptr};

    auto tol_array_ptr = std::make_unique<std::array<ValueType, n_var>>();
    std::fill((*tol_array_ptr).begin(), (*tol_array_ptr).end(),
              tolerance_value);
    tolerance_array = StateContainer<ValueType, n_var>{*tol_array_ptr};
  }
};
