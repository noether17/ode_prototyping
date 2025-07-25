#pragma once

#include <cstring>
#include <vector>

#include "ODEState.hpp"

template <ODEState OutputStateType>
class RawOutput {
 public:
  // StateType used in computation and OutputStateType may differ, e.g.
  // OutputStateType must store data in system memory, but StateType may store
  // data in video memory.
  template <ODEState StateType>
  auto save_state(double t, StateType const& x) {
    times_.push_back(t);
    states_.push_back({});
    x.copy_to_span(span(states_.back()));
  }

  auto const& times() const { return times_; }
  auto const& states() const { return states_; }

 private:
  std::vector<double> times_{};
  std::vector<OutputStateType> states_{};
};
