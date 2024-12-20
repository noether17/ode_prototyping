#pragma once

static constexpr void VDP_ode_kernel(int i, double const* x, double* dxdt) {
  if (i == 0) {
    dxdt[0] = x[1];
  } else if (i == 1) {
    constexpr auto eps = 1.0;
    dxdt[1] = eps * (1.0 - x[0] * x[0]) * x[1] - x[0];
  }
}
