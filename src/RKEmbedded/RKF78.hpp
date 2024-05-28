#pragma once

#include <array>
#include <cmath>
#include <ranges>
#include <vector>

#include "RKEmbedded.hpp"

namespace vws = std::views;

template <typename ODE, typename StateType>
class RKF78 : public RKEmbedded<RKF78<ODE, StateType>, ODE, StateType> {
 public:
  auto static constexpr p = 7;
  auto static constexpr pt = 8;
  auto static constexpr q = std::min(p, pt);
  auto static constexpr a = std::array{
      std::array{2.0 / 27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0},
      std::array{1.0 / 36.0, 1.0 / 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0},
      std::array{1.0 / 24.0, 0.0, 1.0 / 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0},
      std::array{5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0},
      std::array{1.0 / 20.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 5.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0},
      std::array{-25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -65.0 / 27.0,
                 125.0 / 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0,
                 13.0 / 900.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0,
                 67.0 / 90.0, 3.0, 0.0, 0.0, 0.0, 0.0},
      std::array{-91.0 / 108.0, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0,
                 311.0 / 54.0, -19.0 / 60.0, 17.0 / 6.0, -1.0 / 12.0, 0.0, 0.0,
                 0.0},
      std::array{2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0,
                 -301.0 / 82.0, 2133.0 / 4100.0, 45.0 / 82.0, 45.0 / 164.0,
                 18.0 / 41.0, 0.0, 0.0},
      std::array{3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0,
                 -3.0 / 41.0, 3.0 / 41.0, 6.0 / 41.0, 0.0, 0.0},
      std::array{-1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0,
                 -289.0 / 82.0, 2193.0 / 4100.0, 51.0 / 82.0, 33.0 / 164.0,
                 12.0 / 41.0, 0.0, 1.0}};
  auto static constexpr b =
      std::array{41.0 / 840.0, 0.0,        0.0,        0.0,         0.0,
                 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0, 9.0 / 280.0, 9.0 / 280.0,
                 41.0 / 840.0, 0.0,        0.0};
  auto static constexpr bt = std::array{
      0.0,          0.0,          0.0,         0.0,         0.0,
      34.0 / 105.0, 9.0 / 35.0,   9.0 / 35.0,  9.0 / 280.0, 9.0 / 280.0,
      0.0,          41.0 / 840.0, 41.0 / 840.0};
  auto static constexpr db = []() {
    auto db = b;
    for (auto&& [x, xt] : vws::zip(db, bt)) {
      x -= xt;
    }
    return db;
  }();

  auto static constexpr n_stages = static_cast<int>(b.size());
  auto static inline const safety_factor = std::pow(0.38, (1.0 / (1.0 + q)));

  std::array<StateType, n_stages> ks{};
  ODE& ode;
  std::vector<double> times{};
  std::vector<StateType> states{};

  RKF78(ODE& ode) : ode{ode} {}

  auto save_state(double t, StateType const& x) {
    times.push_back(t);
    states.push_back(x);
  }
};
