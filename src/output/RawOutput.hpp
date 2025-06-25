#pragma once

#include <cstring>
#include <vector>

template <typename OutputStateType>
class RawOutput {
 public:
  std::vector<double> times{};
  std::vector<OutputStateType> states{};

  // StateType used in computation and OutputStateType may differ, e.g.
  // OutputStateType must store data in system memory, but StateType may store
  // data in video memory.
  template <typename StateType>
  auto save_state(double t, StateType const& x) {
    times.push_back(t);
    states.push_back({});
    x.copy_to_span(span(states.back()));
  }
};
