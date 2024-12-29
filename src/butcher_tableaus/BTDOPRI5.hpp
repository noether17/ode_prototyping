#pragma once

#include <array>

struct BTDOPRI5 {
  static constexpr auto a = std::array{
      std::array{1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
      std::array{44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0},
      std::array{19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0,
                 -212.0 / 729.0, 0.0, 0.0},
      std::array{9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
                 -5103.0 / 18656.0, 0.0},
      std::array{35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
                 -2187.0 / 6784.0, 11.0 / 84.0}};
  static constexpr auto b = std::array{
      35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
      11.0 / 84.0,  0.0};
  static constexpr auto bt = std::array{5179.0 / 57600.0,    0.0,
                                        7571.0 / 16695.0,    393.0 / 640.0,
                                        -92097.0 / 339200.0, 187.0 / 2100.0,
                                        1.0 / 40.0};
  static constexpr auto p = 5;
  static constexpr auto pt = 4;
  static constexpr auto n_stages = static_cast<int>(b.size());
};
