#pragma once

#include <array>

#include "RKEmbeddedCuda.cuh"  // for block_size

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
struct CudaNBodyOde {
  auto static constexpr n_particles = n_var / 6;
  auto static constexpr n_pairs = n_particles * (n_particles - 1) / 2;
  auto static constexpr dim = 3;
  std::array<double, n_particles> masses;
  void compute_rhs(double const* x, double* f) {
    cudaMemcpy(f, x + n_var / 2, (n_var / 2) * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemset(f + n_var / 2, 0, (n_var / 2) * sizeof(double));
    cuda_n_body_acc_kernel<decltype(masses)>
        <<<num_blocks<n_pairs>(), block_size>>>(x, f + n_var / 2, masses,
                                                n_pairs, n_particles);
  }

  CudaNBodyOde(std::array<double, n_particles> const& masses)
      : masses{masses} {}
};
