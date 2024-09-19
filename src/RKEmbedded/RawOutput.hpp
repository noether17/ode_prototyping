#pragma once

#include <cmath>
#include <vector>

template <typename StateType>
class RawOutput {
 public:
  std::vector<double> times{};
  std::vector<StateType> states{};

  auto save_state(double t, StateType const& x) {
    times.push_back(t);
    states.push_back(x);
  }
};
