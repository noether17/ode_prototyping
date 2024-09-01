#pragma once

#include <array>

struct BTRKF45 {
  auto static constexpr a = std::array{
      std::array{1.0 / 4.0, 0.0, 0.0, 0.0, 0.0},
      std::array{3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0},
      std::array{1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0},
      std::array{439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0},
      std::array{-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0,
                 -11.0 / 40.0}};
  auto static constexpr b = std::array{25.0 / 216.0,    0.0,  1408.0 / 2565.0,
                                       2197.0 / 4104.0, -0.2, 0.0};
  auto static constexpr bt =
      std::array{16.0 / 135.0,      0.0,         6656.0 / 12825.0,
                 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0};
  auto static constexpr p = 4;
  auto static constexpr pt = 5;
  auto static constexpr n_stages = static_cast<int>(b.size());
};
