#pragma once

static constexpr auto exp_ode_kernel(int i, double const* x, double* dxdt) {
  dxdt[i] = x[i];
}
