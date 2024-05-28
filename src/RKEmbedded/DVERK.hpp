#pragma once

#include <array>
#include <cmath>
#include <ranges>
#include <vector>

#include "RKEmbedded.hpp"

namespace vws = std::views;

/* This method performs poorly and gets wrong result for simple exponential ODE.
 * Look into this further. */
template <typename ODE, typename StateType>
class DVERK : public RKEmbedded<DVERK<ODE, StateType>, ODE, StateType> {
 public:
  auto static constexpr p = 6;
  auto static constexpr pt = 5;
  auto static constexpr q = std::min(p, pt);
  auto static constexpr a = std::array{
      std::array{1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{4.0 / 75.0, 16.0 / 75.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{5.0 / 6.0, -8.0 / 3.0, 5.0 / 2.0, 0.0, 0.0, 0.0, 0.0},
      std::array{-165.0 / 64.0, 55.0 / 6.0, -425.0 / 64.0, 85.0 / 96.0, 0.0,
                 0.0, 0.0},
      std::array{12.0 / 5.0, -8.0, 4015.0 / 612.0, -11.0 / 36.0, 88.0 / 255.0,
                 0.0, 0.0},
      std::array{-8263.0 / 15000.0, 124.0 / 75.0, -643.0 / 680.0, -81.0 / 250.0,
                 2484.0 / 10625.0, 0.0, 0.0},
      std::array{3501.0 / 1720.0, -300.0 / 43.0, 297275.0 / 52632.0,
                 -319.0 / 2322.0, 24068.0 / 84065.0, 0.0, 3850.0 / 26703.0}};
  auto static constexpr b =
      std::array{3.0 / 40.0,     0.0, 875.0 / 2244.0,  23.0 / 72.0,
                 264.0 / 1955.0, 0.0, 125.0 / 11592.0, 43.0 / 616.0};
  auto static constexpr bt = std::array{
      13.0 / 160.0, 0.0, 2375.0 / 5984.0, 5.0 / 16.0, 12.0 / 85.0, 3.0 / 44.0,
      0.0,          0.0};
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

  DVERK(ODE& ode) : ode{ode} {}

  auto save_state(double t, StateType const& x) {
    times.push_back(t);
    states.push_back(x);
  }
};
