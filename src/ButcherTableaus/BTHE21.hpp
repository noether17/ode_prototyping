#pragma once

#include <array>

struct BTHE21 {
  auto static constexpr a = std::array<std::array<double, 1>, 1>{{1.0}};
  auto static constexpr b = std::array{1.0 / 2.0, 1.0 / 2.0};
  auto static constexpr bt = std::array{1.0, 0.0};
  auto static constexpr p = 2;
  auto static constexpr pt = 1;
  auto static constexpr n_stages = static_cast<int>(b.size());
};
