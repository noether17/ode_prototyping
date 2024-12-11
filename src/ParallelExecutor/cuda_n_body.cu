#include <array>
#include <fstream>
#include <span>
#include <vector>

#include "BTRKF78.hpp"
#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "RKEmbeddedParallel.hpp"
#include "RawOutput.hpp"

template <typename CudaStateT>
struct RawCudaOutput {
  std::vector<double> times{};
  std::vector<std::vector<double>> states{};

  void save_state(double t, CudaStateT const& x) {
    auto host_x = std::vector<double>(x.size());
    cudaMemcpy(host_x.data(), x.data(), x.size() * sizeof(double),
               cudaMemcpyDeviceToHost);
    times.push_back(t);
    states.push_back(host_x);
  }
};

struct ODENBody {
  static constexpr auto n_particles = 5;
  static constexpr auto n_var = n_particles * 6;
  // static inline std::array<double, n_particles> const
  // masses{1.0, 1.0, 1.0, 1.0,
  //                                                            1.0};
  // static double* dev_masses;
  // ODENBody() {
  //   cudaMalloc(&dev_masses, n_particles * sizeof(double));
  //   cudaMemcpy(dev_masses, masses.data(), n_particles * sizeof(double),
  //              cudaMemcpyHostToDevice);
  // }
  //~ODENBody() { cudaFree(dev_masses); }
  static __device__ void ode_kernel(int i, double const* x, double* dxdt) {
    for (auto j = i + 1; j < n_particles; ++j) {
      auto dx = x[3 * j] - x[3 * i];
      auto dy = x[3 * j + 1] - x[3 * i + 1];
      auto dz = x[3 * j + 2] - x[3 * i + 2];
      auto dist = std::sqrt(dx * dx + dy * dy + dz * dz);
      auto dist_3 = dist * dist * dist;
      auto ax = dx / dist_3;
      auto ay = dy / dist_3;
      auto az = dz / dist_3;
      atomicAdd(&dxdt[3 * i], ax);
      atomicAdd(&dxdt[3 * i + 1], ay);
      atomicAdd(&dxdt[3 * i + 2], az);
      atomicAdd(&dxdt[3 * j], -ax);
      atomicAdd(&dxdt[3 * j + 1], -ay);
      atomicAdd(&dxdt[3 * j + 2], -az);
    }
  }
  void operator()(auto& exe, std::span<double const, n_var> x, double* dxdt) {
    constexpr auto vel_offset = n_var / 2;
    cudaMemcpy(dxdt, x.data() + vel_offset, vel_offset * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemset(dxdt + vel_offset, 0, vel_offset * sizeof(double));
    exe.template call_parallel_kernel<ode_kernel>(n_particles, x.data(),
                                                  dxdt + vel_offset);
  }
};

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

  auto ode_n_body = ODENBody{};
  auto cuda_exe = CudaExecutor{};
  auto integrator =
      RKEmbeddedParallel<CudaState, double, n_var, BTRKF78, ODENBody,
                         RawCudaOutput<CudaState<double, n_var>>,
                         CudaExecutor>{};
  auto output = RawCudaOutput<CudaState<double, n_var>>{};

  auto x0 = CudaState<double, n_var>{x0_data};
  auto t0 = 0.0;
  auto tf = 6.3;
  auto host_tol = std::array<double, n_var>{};
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-10);
  auto tol = CudaState<double, n_var>{host_tol};

  integrator.integrate(x0, t0, tf, tol, tol, ode_n_body, output, cuda_exe);

  auto output_file = std::ofstream{"RKF78_cuda_n_body_output.txt"};
  for (std::size_t i = 0; i < output.times.size(); ++i) {
    output_file << output.times[i];
    for (std::size_t j = 0; j < n_var; ++j) {
      output_file << ',' << output.states[i][j];
    }
    output_file << '\n';
  }

  return 0;
}
