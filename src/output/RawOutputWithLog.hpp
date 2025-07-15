#pragma once

#include <cstring>
#include <iomanip>
#include <ios>
#include <iostream>
#include <vector>

#include "ODEState.hpp"

template <ODEState OutputStateType>
class RawOutputWithLog {
 public:
  ~RawOutputWithLog() { std::cout << '\n'; }

  // StateType used in computation and OutputStateType may differ, e.g.
  // OutputStateType must store data in system memory, but StateType may store
  // data in video memory.
  template <ODEState StateType>
  auto save_state(double t, StateType const& x) {
    auto cout_flags = std::cout.flags();
    std::cout << "\r      Saving state " << times_.size()
              << " at t=" << std::setprecision(3) << std::scientific << t
              << std::flush;
    std::cout.flags(cout_flags);

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
