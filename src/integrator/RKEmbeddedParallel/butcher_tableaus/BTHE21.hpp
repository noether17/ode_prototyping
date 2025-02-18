#pragma once

#include <array>

struct BTHE21 {
  static constexpr auto a = std::array<std::array<double, 1>, 1>{{1.0}};
  static constexpr auto b = std::array{1.0 / 2.0, 1.0 / 2.0};
  static constexpr auto bt = std::array{1.0, 0.0};
  static constexpr auto p = 2;
  static constexpr auto pt = 1;
  static constexpr auto n_stages = static_cast<int>(b.size());
};
