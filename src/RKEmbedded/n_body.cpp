#include <array>
#include <fstream>

#include "AllocatedState.hpp"
#include "RKF78.hpp"

int main() {
  auto constexpr n_var = 12;
  auto ode_n_body = [](AllocatedState<n_var> const& x,
                       AllocatedState<n_var>& dxdt) {
    auto dx = x[3] - x[0];
    auto dy = x[4] - x[1];
    auto dz = x[5] - x[2];
    auto dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    auto dist_3 = dist * dist * dist;
    auto ax = dx / dist_3;
    auto ay = dy / dist_3;
    auto az = dz / dist_3;
    dxdt[0] = x[6];
    dxdt[1] = x[7];
    dxdt[2] = x[8];
    dxdt[3] = x[9];
    dxdt[4] = x[10];
    dxdt[5] = x[11];
    dxdt[6] = ax;
    dxdt[7] = ay;
    dxdt[8] = az;
    dxdt[9] = -ax;
    dxdt[10] = -ay;
    dxdt[11] = -az;
  };

  auto integrator =
      RKF78<decltype(ode_n_body), AllocatedState<n_var>>{ode_n_body};

  auto x0_data =
      std::array{1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0};
  auto x0 = AllocatedState<n_var>{x0_data};
  auto t0 = 0.0;
  auto tf = 10.0;
  auto tol = AllocatedState<n_var>{};
  fill(tol, 1.0e-3);

  integrator.integrate(x0, t0, tf, tol, tol);

  auto output_file = std::ofstream{"RKF78_n_body_output.txt"};
  for (std::size_t i = 0; i < integrator.times.size(); ++i) {
    output_file << integrator.times[i];
    for (std::size_t j = 0; j < n_var; ++j) {
      output_file << ',' << integrator.states[i][j];
    }
    output_file << '\n';
  }

  return 0;
}
