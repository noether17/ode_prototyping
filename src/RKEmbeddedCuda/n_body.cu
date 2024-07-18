#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "RKEmbeddedCuda.cuh"

template <typename MassType>
__global__ void cuda_n_body_acc_kernel(double const* x, double* a,
                                       MassType masses, int n_pairs,
                                       int n_particles) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto n_1_2 = n_particles - 0.5;
  while (tid < n_pairs) {
    auto i = static_cast<int>(n_1_2 - std::sqrt(n_1_2 * n_1_2 - 2.0 * tid));
    auto j = tid - (n_particles - 1) * i + (i * (i + 1)) / 2 + 1;
    auto ix = x[3 * i];
    auto iy = x[3 * i + 1];
    auto iz = x[3 * i + 2];
    auto jx = x[3 * j];
    auto jy = x[3 * j + 1];
    auto jz = x[3 * j + 2];
    auto dx = jx - ix;
    auto dy = jy - iy;
    auto dz = jz - iz;
    auto dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    auto dist_3 = dist * dist * dist;
    auto ax = dx / dist_3;
    auto ay = dy / dist_3;
    auto az = dz / dist_3;
    atomicAdd(&a[3 * i], ax * masses[j]);
    atomicAdd(&a[3 * i + 1], ay * masses[j]);
    atomicAdd(&a[3 * i + 2], az * masses[j]);
    atomicAdd(&a[3 * j], -ax * masses[i]);
    atomicAdd(&a[3 * j + 1], -ay * masses[i]);
    atomicAdd(&a[3 * j + 2], -az * masses[i]);
    tid += blockDim.x * gridDim.x;
  }
}

template <int n_var>
struct CUDANBodyODE {
  auto static constexpr n_particles = n_var / 6;
  auto static constexpr n_pairs = n_particles * (n_particles - 1) / 2;
  auto static constexpr dim = 3;
  auto static constexpr masses = std::array{1.0, 1.0, 1.0, 1.0, 1.0};
  static void compute_rhs(double const* x, double* f) {
    cudaMemcpy(f, x + n_var / 2, (n_var / 2) * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemset(f + n_var / 2, 0, (n_var / 2) * sizeof(double));
    cuda_n_body_acc_kernel<decltype(masses)>
        <<<num_blocks<n_pairs>(), block_size>>>(x, f + n_var / 2, masses,
                                                n_pairs, n_particles);
  }
};

template <int n_var>
struct RawCudaOutput {
  std::vector<double> times{};
  std::vector<std::vector<double>> states{};

  void save_state(double t, double const* x_ptr) {
    auto host_x = std::vector<double>(n_var);
    cudaMemcpy(host_x.data(), x_ptr, n_var * sizeof(double),
               cudaMemcpyDeviceToHost);
    times.push_back(t);
    states.push_back(host_x);
  }
};

int main() {
  // Simple Two-Body Orbit
  // auto host_x0 =
  //     std::array{1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5,
  //     0.0};
  // Three-Body Figure-8
  // auto host_x0 =
  //    std::array{0.9700436,   -0.24308753, 0.0, -0.9700436,  0.24308753,  0.0,
  //               0.0,         0.0,         0.0, 0.466203685, 0.43236573,  0.0,
  //               0.466203685, 0.43236573,  0.0, -0.93240737, -0.86473146,
  //               0.0};
  // Pythagorean Three-Body
  // auto host_x0 = std::array{1.0, 3.0, 0.0, -2.0, -1.0, 0.0, 1.0, -1.0, 0.0,
  //                           0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0,  0.0};
  // Five-Body Double Figure-8
  auto host_x0 =
      std::array{1.657666,  0.0,       0.0, 0.439775,  -0.169717, 0.0,
                 -1.268608, -0.267651, 0.0, -1.268608, 0.267651,  0.0,
                 0.439775,  0.169717,  0.0, 0.0,       -0.593786, 0.0,
                 1.822785,  0.128248,  0.0, 1.271564,  0.168645,  0.0,
                 -1.271564, 0.168645,  0.0, -1.822785, 0.128248,  0.0};
  auto constexpr n_var = host_x0.size();
  auto t0 = 0.0;
  auto tf = 6.3;
  auto host_tol = std::array<double, n_var>{};
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-10);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto ode = CUDANBodyODE<n_var>{};
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, RKF78, CUDANBodyODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  auto output_file = std::ofstream{"n_body_output.txt"};
  for (auto i = 0; i < output.times.size(); ++i) {
    output_file << output.times[i];
    for (auto j = 0; j < n_var; ++j) {
      output_file << ',' << output.states[i][j];
    }
    output_file << '\n';
  }

  cudaFree(dev_tol);
  cudaFree(dev_x0);

  return 0;
}
