#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "RKEmbedded.hpp"

template <typename ODE, typename StateType>
class HE21 : public RKEmbedded<HE21<ODE, StateType>, ODE, StateType> {
 public:
  auto static constexpr p = 2;
  auto static constexpr pt = 1;
  auto static constexpr a = std::array<std::array<double, 1>, 1>{{1.0}};
  auto static constexpr b = std::array{1.0 / 2.0, 1.0 / 2.0};
  auto static constexpr bt = std::array{1.0, 0.0};

  auto static constexpr n_stages = static_cast<int>(b.size());

  std::array<StateType, n_stages> ks{};
  ODE& ode;
  std::vector<double> times{};
  std::vector<StateType> states{};

  HE21(ODE& ode) : ode{ode} {}

  auto save_state(double t, StateType const& x) {
    times.push_back(t);
    states.push_back(x);
  }
};
