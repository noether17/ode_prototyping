#pragma once

#include <array>

#include "RKEmbeddedCuda.cuh"  // for block_size

template <typename MassType>
__global__ void cuda_n_body_acc_kernel_with_masses(double const* x, double* a,
                                                   MassType masses, int n_pairs,
                                                   int n_particles,
                                                   double softening_2) {
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
    auto dist_sq = dx * dx + dy * dy + dz * dz;
    auto dist = std::sqrt(dist_sq);
    auto denominator = dist * (dist_sq + softening_2);
    auto ax = dx / denominator;
    auto ay = dy / denominator;
    auto az = dz / denominator;
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
struct CudaNBodyOdeWithMasses {
  auto static constexpr n_particles = n_var / 6;
  auto static constexpr n_pairs = n_particles * (n_particles - 1) / 2;
  auto static constexpr dim = 3;
  std::array<double, n_particles> masses{};
  double softening_2{};
  void compute_rhs(double const* x, double* f) {
    cudaMemcpy(f, x + n_var / 2, (n_var / 2) * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemset(f + n_var / 2, 0, (n_var / 2) * sizeof(double));
    cuda_n_body_acc_kernel_with_masses<decltype(masses)>
        <<<num_blocks<n_pairs>(), block_size>>>(
            x, f + n_var / 2, masses, n_pairs, n_particles, softening_2);
  }

  CudaNBodyOdeWithMasses(std::array<double, n_particles> const& ms,
                         double softening = 0.0)
      : masses{ms}, softening_2{softening * softening} {}
};

__global__ void cuda_n_body_acc_kernel(double const* x, double* a, int n_pairs,
                                       int n_particles, double softening_2) {
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
    auto dist_sq = dx * dx + dy * dy + dz * dz;
    auto dist = std::sqrt(dist_sq);
    auto denominator = dist * (dist_sq + softening_2);
    auto ax = dx / denominator;
    auto ay = dy / denominator;
    auto az = dz / denominator;
    atomicAdd(&a[3 * i], ax * 1.0);
    atomicAdd(&a[3 * i + 1], ay * 1.0);
    atomicAdd(&a[3 * i + 2], az * 1.0);
    atomicAdd(&a[3 * j], -ax * 1.0);
    atomicAdd(&a[3 * j + 1], -ay * 1.0);
    atomicAdd(&a[3 * j + 2], -az * 1.0);
    tid += blockDim.x * gridDim.x;
  }
}

template <int n_var>
struct CudaNBodyOde {
  auto static constexpr n_particles = n_var / 6;
  auto static constexpr n_pairs = n_particles * (n_particles - 1) / 2;
  auto static constexpr dim = 3;

  double softening_2{};
  void compute_rhs(double const* x, double* f) {
    cudaMemcpy(f, x + n_var / 2, (n_var / 2) * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemset(f + n_var / 2, 0, (n_var / 2) * sizeof(double));
    cuda_n_body_acc_kernel<<<num_blocks<n_pairs>(), block_size>>>(
        x, f + n_var / 2, n_pairs, n_particles, softening_2);
  }

  CudaNBodyOde(double softening = 0.0) : softening_2{softening * softening} {}
};

__global__ void cuda_n_body_acc_kernel_simple(double const* x, double* a,
                                              int n_particles,
                                              double softening_sq) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_particles) {
    a[3 * i] = 0.0;
    a[3 * i + 1] = 0.0;
    a[3 * i + 2] = 0.0;
    for (int j = 0; j < n_particles; ++j) {
      if (i == j) {
        continue;
      }
      auto dx = x[3 * j] - x[3 * i];
      auto dy = x[3 * j + 1] - x[3 * i + 1];
      auto dz = x[3 * j + 2] - x[3 * i + 2];
      auto dist_sq = dx * dx + dy * dy + dz * dz;
      auto dist = std::sqrt(dist_sq);
      auto denominator = dist * (dist_sq + softening_sq);
      a[3 * i] += dx / denominator;
      a[3 * i + 1] += dy / denominator;
      a[3 * i + 2] += dy / denominator;
    }
    i += blockDim.x * gridDim.x;
  }
}

template <int n_var>
struct CudaNBodyOdeSimple {
  auto static constexpr n_particles = n_var / 6;

  double softening_sq{};
  void compute_rhs(double const* x, double* f) {
    cudaMemcpy(f, x + n_var / 2, (n_var / 2) * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemset(f + n_var / 2, 0, (n_var / 2) * sizeof(double));
    cuda_n_body_acc_kernel_simple<<<num_blocks<n_particles>(), block_size>>>(
        x, f + n_var / 2, n_particles, softening_sq);
  }

  CudaNBodyOdeSimple(double softening = 0.0)
      : softening_sq{softening * softening} {}
};
