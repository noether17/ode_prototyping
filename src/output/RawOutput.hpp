#pragma once

#include <cstring>
#include <vector>

#include "State.hpp"

template <typename OutputStateType, typename InputStateType>
auto copy_out(InputStateType const& x) {
  auto output_state = OutputStateType{};
  std::memcpy(
      output_state.data(), x.data(),
      std::size(x) * sizeof(typename state_traits<InputStateType>::value_type));
  return output_state;
}

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
    states.push_back(copy_out<OutputStateType>(x));
  }
};
