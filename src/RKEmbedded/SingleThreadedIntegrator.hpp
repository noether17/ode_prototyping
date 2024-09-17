#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "RKEmbedded.hpp"

template <typename ButcherTableau, typename ODE, typename StateType>
class SingleThreadedIntegrator
    : public RKEmbedded<
          SingleThreadedIntegrator<ButcherTableau, ODE, StateType>, ODE,
          StateType>,
      ButcherTableau {
 public:
  using ButcherTableau::a;
  using ButcherTableau::b;
  using ButcherTableau::bt;
  using ButcherTableau::n_stages;
  using ButcherTableau::p;
  using ButcherTableau::pt;

  std::array<StateType, n_stages> ks{};
  ODE& ode;
  std::vector<double> times{};
  std::vector<StateType> states{};

  SingleThreadedIntegrator(ODE& ode) : ode{ode} {}

  auto save_state(double t, StateType const& x) {
    times.push_back(t);
    states.push_back(x);
  }
};
