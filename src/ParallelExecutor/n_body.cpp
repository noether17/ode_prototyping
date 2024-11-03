#include <array>
#include <fstream>

#include "AllocatedState.hpp"
#include "BTRKF78.hpp"
#include "ParallelThreadPool.hpp"
#include "RKEmbeddedParallel.hpp"
#include "RawOutput.hpp"

int main() {
  // auto x0_data =
  //     std::array{1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5,
  //     0.0};
  // auto x0_data =
  //    std::array{0.9700436,   -0.24308753, 0.0, -0.9700436,  0.24308753,  0.0,
  //               0.0,         0.0,         0.0, 0.466203685, 0.43236573,  0.0,
  //               0.466203685, 0.43236573,  0.0, -0.93240737, -0.86473146,
  //               0.0};
  // auto x0_data = std::array{1.0, 3.0, 0.0, -2.0, -1.0, 0.0, 1.0, -1.0, 0.0,
  //                          0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0,  0.0};
  auto x0_data =
      std::array{1.657666,  0.0,       0.0, 0.439775,  -0.169717, 0.0,
                 -1.268608, -0.267651, 0.0, -1.268608, 0.267651,  0.0,
                 0.439775,  0.169717,  0.0, 0.0,       -0.593786, 0.0,
                 1.822785,  0.128248,  0.0, 1.271564,  0.168645,  0.0,
                 -1.271564, 0.168645,  0.0, -1.822785, 0.128248,  0.0};
  auto masses = std::array{1.0, 1.0, 1.0, 1.0, 1.0};
  auto constexpr n_var = x0_data.size();

  auto ode_n_body = [masses](AllocatedState<n_var> const& x,
                             AllocatedState<n_var>& dxdt) {
    auto constexpr vel_offset = n_var / 2;
    for (std::size_t i = 0; i < vel_offset; ++i) {
      dxdt[i] = x[i + vel_offset];
      dxdt[i + vel_offset] = 0.0;
    }
    auto constexpr n_particles = n_var / 6;
    for (std::size_t i = 0; i < n_particles; ++i) {
      for (auto j = i + 1; j < n_particles; ++j) {
        auto dx = x[3 * j] - x[3 * i];
        auto dy = x[3 * j + 1] - x[3 * i + 1];
        auto dz = x[3 * j + 2] - x[3 * i + 2];
        auto dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        auto dist_3 = dist * dist * dist;
        auto ax = dx / dist_3;
        auto ay = dy / dist_3;
        auto az = dz / dist_3;
        dxdt[vel_offset + 3 * i] += ax * masses[j];
        dxdt[vel_offset + 3 * i + 1] += ay * masses[j];
        dxdt[vel_offset + 3 * i + 2] += az * masses[j];
        dxdt[vel_offset + 3 * j] += -ax * masses[i];
        dxdt[vel_offset + 3 * j + 1] += -ay * masses[i];
        dxdt[vel_offset + 3 * j + 2] += -az * masses[i];
      }
    }
  };
  auto thread_pool = ParallelThreadPool(8);
  auto integrator =
      RKEmbeddedParallel<AllocatedState<n_var>, BTRKF78, decltype(ode_n_body),
                         RawOutput<AllocatedState<n_var>>,
                         ParallelThreadPool>{};
  auto output = RawOutput<AllocatedState<n_var>>{};

  auto x0 = AllocatedState<n_var>{x0_data};
  auto t0 = 0.0;
  auto tf = 6.3;
  auto tol = AllocatedState<n_var>{};
  fill(tol, 1.0e-10);

  integrator.integrate(x0, t0, tf, tol, tol, ode_n_body, output, thread_pool);

  auto output_file = std::ofstream{"RKF78_parallel_n_body_output.txt"};
  for (std::size_t i = 0; i < output.times.size(); ++i) {
    output_file << output.times[i];
    for (std::size_t j = 0; j < n_var; ++j) {
      output_file << ',' << output.states[i][j];
    }
    output_file << '\n';
  }

  return 0;
}
