#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "BTHE21.hpp"
#include "CUDAExpODE.cuh"
#include "HostUtils.hpp"
#include "RKEmbeddedCuda.cuh"

TEST(RKEmbeddedCudaTest, ComputeErrorTargetTestSmall) {
  auto constexpr n_var = 10;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_rtol = std::vector<double>(n_var);
  std::iota(host_rtol.begin(), host_rtol.end(), 1.0);
  auto host_atol = std::vector<double>(n_var);
  std::iota(host_atol.begin(), host_atol.end(), 2.0);
  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  cudaMemcpy(dev_x, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_rtol = nullptr;
  cudaMalloc(&dev_rtol, n_var * sizeof(double));
  cudaMemcpy(dev_rtol, host_rtol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_atol = nullptr;
  cudaMalloc(&dev_atol, n_var * sizeof(double));
  cudaMemcpy(dev_atol, host_atol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_error_target = nullptr;
  cudaMalloc(&dev_error_target, n_var * sizeof(double));

  cuda_compute_error_target<<<num_blocks<n_var>(), block_size>>>(
      dev_x, dev_rtol, dev_atol, dev_error_target, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_error_target, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_atol[i] + host_rtol[i] * std::abs(host_x[i]);
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_error_target);
  cudaFree(dev_atol);
  cudaFree(dev_rtol);
  cudaFree(dev_x);
}

TEST(RKEmbeddedCudaTest, ComputeErrorTargetTestLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_rtol = std::vector<double>(n_var);
  std::iota(host_rtol.begin(), host_rtol.end(), 1.0);
  auto host_atol = std::vector<double>(n_var);
  std::iota(host_atol.begin(), host_atol.end(), 2.0);
  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  cudaMemcpy(dev_x, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_rtol = nullptr;
  cudaMalloc(&dev_rtol, n_var * sizeof(double));
  cudaMemcpy(dev_rtol, host_rtol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_atol = nullptr;
  cudaMalloc(&dev_atol, n_var * sizeof(double));
  cudaMemcpy(dev_atol, host_atol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_error_target = nullptr;
  cudaMalloc(&dev_error_target, n_var * sizeof(double));

  cuda_compute_error_target<<<num_blocks<n_var>(), block_size>>>(
      dev_x, dev_rtol, dev_atol, dev_error_target, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_error_target, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_atol[i] + host_rtol[i] * std::abs(host_x[i]);
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_error_target);
  cudaFree(dev_atol);
  cudaFree(dev_rtol);
  cudaFree(dev_x);
}

TEST(RKEmbeddedCudaTest, RKNormTestSmall) {
  auto constexpr n_var = 10;
  auto host_v = std::vector<double>(n_var);
  std::iota(host_v.begin(), host_v.end(), 0.0);
  auto host_scale = std::vector<double>(n_var);
  std::iota(host_scale.begin(), host_scale.end(), 1.0);
  double* dev_v = nullptr;
  cudaMalloc(&dev_v, n_var * sizeof(double));
  cudaMemcpy(dev_v, host_v.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_scale = nullptr;
  cudaMalloc(&dev_scale, n_var * sizeof(double));
  cudaMemcpy(dev_scale, host_scale.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);

  auto host_cuda_result = cuda_rk_norm<n_var>(dev_v, dev_scale);

  auto host_result = std::sqrt(
      std::inner_product(host_v.begin(), host_v.end(), host_scale.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      n_var);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);

  cudaFree(dev_scale);
  cudaFree(dev_v);
}

TEST(RKEmbeddedCudaTest, RKNormTestLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_v = std::vector<double>(n_var);
  std::iota(host_v.begin(), host_v.end(), 0.0);
  auto host_scale = std::vector<double>(n_var);
  std::iota(host_scale.begin(), host_scale.end(), 1.0);
  double* dev_v = nullptr;
  cudaMalloc(&dev_v, n_var * sizeof(double));
  cudaMemcpy(dev_v, host_v.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_scale = nullptr;
  cudaMalloc(&dev_scale, n_var * sizeof(double));
  cudaMemcpy(dev_scale, host_scale.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);

  auto host_cuda_result = cuda_rk_norm<n_var>(dev_v, dev_scale);

  auto host_result = std::sqrt(
      std::inner_product(host_v.begin(), host_v.end(), host_scale.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      n_var);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);

  cudaFree(dev_scale);
  cudaFree(dev_v);
}

TEST(RKEmbeddedCudaTest, EulerStepSmall) {
  auto constexpr n_var = 10;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_f = std::vector<double>(n_var);
  std::iota(host_f.begin(), host_f.end(), 1.0);
  auto dt = 0.1;
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_f0 = nullptr;
  cudaMalloc(&dev_f0, n_var * sizeof(double));
  cudaMemcpy(dev_f0, host_f.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_x1 = nullptr;
  cudaMalloc(&dev_x1, n_var * sizeof(double));

  cuda_euler_step<<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_f0, dt,
                                                       dev_x1, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_x1, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_x[i] + dt * host_f[i];
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_x1);
  cudaFree(dev_f0);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, EulerStepLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_f = std::vector<double>(n_var);
  std::iota(host_f.begin(), host_f.end(), 1.0);
  auto dt = 0.1;
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_f0 = nullptr;
  cudaMalloc(&dev_f0, n_var * sizeof(double));
  cudaMemcpy(dev_f0, host_f.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_x1 = nullptr;
  cudaMalloc(&dev_x1, n_var * sizeof(double));

  cuda_euler_step<<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_f0, dt,
                                                       dev_x1, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_x1, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_x[i] + dt * host_f[i];
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_x1);
  cudaFree(dev_f0);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, EstimateInitialStepSmall) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto host_atol = std::vector<double>(n_var);
  std::iota(host_atol.begin(), host_atol.end(), 1.0);
  auto host_rtol = std::vector<double>(n_var);
  std::iota(host_rtol.begin(), host_rtol.end(), 2.0);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_atol = nullptr;
  cudaMalloc(&dev_atol, n_var * sizeof(double));
  cudaMemcpy(dev_atol, host_atol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_rtol = nullptr;
  cudaMalloc(&dev_rtol, n_var * sizeof(double));
  cudaMemcpy(dev_rtol, host_rtol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);

  auto ode = CUDAExpODE<n_var>{};
  auto host_cuda_result =
      cuda_estimate_initial_step<n_var, BTHE21, CUDAExpODE<n_var>>(
          dev_x0, dev_atol, dev_rtol, ode);

  auto host_result =
      host_estimate_initial_step<BTHE21>(host_x0, host_atol, host_rtol);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);

  cudaFree(dev_rtol);
  cudaFree(dev_atol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, EstimateInitialStepLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto host_atol = std::vector<double>(n_var);
  std::iota(host_atol.begin(), host_atol.end(), 1.0);
  auto host_rtol = std::vector<double>(n_var);
  std::iota(host_rtol.begin(), host_rtol.end(), 2.0);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_atol = nullptr;
  cudaMalloc(&dev_atol, n_var * sizeof(double));
  cudaMemcpy(dev_atol, host_atol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_rtol = nullptr;
  cudaMalloc(&dev_rtol, n_var * sizeof(double));
  cudaMemcpy(dev_rtol, host_rtol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);

  auto ode = CUDAExpODE<n_var>{};
  auto host_cuda_result =
      cuda_estimate_initial_step<n_var, BTHE21, CUDAExpODE<n_var>>(
          dev_x0, dev_atol, dev_rtol, ode);

  auto host_result =
      host_estimate_initial_step<BTHE21>(host_x0, host_atol, host_rtol);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);

  cudaFree(dev_rtol);
  cudaFree(dev_atol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, RKStagesSmall) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto dt = 0.1;
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_ks = nullptr;
  cudaMalloc(&dev_ks, n_var * BTHE21::n_stages * sizeof(double));
  double* dev_temp_state = nullptr;
  cudaMalloc(&dev_temp_state, n_var * sizeof(double));

  auto ode = CUDAExpODE<n_var>{};
  cuda_evaluate_stages<n_var, BTHE21, CUDAExpODE<n_var>>(dev_x0, dev_temp_state,
                                                         dev_ks, dt, ode);

  auto host_cuda_result = std::vector<double>(n_var * BTHE21::n_stages);
  cudaMemcpy(host_cuda_result.data(), dev_ks,
             n_var * BTHE21::n_stages * sizeof(double), cudaMemcpyDeviceToHost);
  auto host_temp_state = std::vector<double>(n_var);
  auto host_result = std::vector<double>(n_var * BTHE21::n_stages);
  host_evaluate_stages<n_var, BTHE21, HostExpODE<n_var>>(
      host_x0.data(), host_temp_state.data(), host_result.data(), dt);
  for (auto i = 0; i < n_var * BTHE21::n_stages; ++i) {
    EXPECT_DOUBLE_EQ(host_result[i], host_cuda_result[i]);
  }

  cudaFree(dev_temp_state);
  cudaFree(dev_ks);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, RKStagesLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto dt = 0.1;
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_ks = nullptr;
  cudaMalloc(&dev_ks, n_var * BTHE21::n_stages * sizeof(double));
  double* dev_temp_state = nullptr;
  cudaMalloc(&dev_temp_state, n_var * sizeof(double));

  auto ode = CUDAExpODE<n_var>{};
  cuda_evaluate_stages<n_var, BTHE21, CUDAExpODE<n_var>>(dev_x0, dev_temp_state,
                                                         dev_ks, dt, ode);

  auto host_cuda_result = std::vector<double>(n_var * BTHE21::n_stages);
  cudaMemcpy(host_cuda_result.data(), dev_ks,
             n_var * BTHE21::n_stages * sizeof(double), cudaMemcpyDeviceToHost);
  auto host_temp_state = std::vector<double>(n_var);
  auto host_result = std::vector<double>(n_var * BTHE21::n_stages);
  host_evaluate_stages<n_var, BTHE21, HostExpODE<n_var>>(
      host_x0.data(), host_temp_state.data(), host_result.data(), dt);
  for (auto i = 0; i < n_var * BTHE21::n_stages; ++i) {
    EXPECT_DOUBLE_EQ(host_result[i], host_cuda_result[i]);
  }

  cudaFree(dev_temp_state);
  cudaFree(dev_ks);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, UpdateStateAndErrorSmall) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto host_ks = std::vector<double>(n_var * BTHE21::n_stages);
  std::iota(host_ks.begin(), host_ks.end(), 1.0);
  auto dt = 0.1;
  auto b = BTHE21::b;
  auto db = []() {
    auto db = BTHE21::b;
    for (auto i = 0; i < BTHE21::n_stages; ++i) {
      db[i] -= BTHE21::bt[i];
    }
    return db;
  }();
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_ks = nullptr;
  cudaMalloc(&dev_ks, n_var * BTHE21::n_stages * sizeof(double));
  cudaMemcpy(dev_ks, host_ks.data(), n_var * BTHE21::n_stages * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  double* dev_error_estimate = nullptr;
  cudaMalloc(&dev_error_estimate, n_var * sizeof(double));

  cuda_update_state_and_error<n_var, BTHE21>
      <<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_ks, dt, dev_x,
                                            dev_error_estimate, b, db,
                                            BTHE21::n_stages);

  auto host_cuda_x = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_x.data(), dev_x, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  auto host_cuda_error_estimate = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_error_estimate.data(), dev_error_estimate,
             n_var * sizeof(double), cudaMemcpyDeviceToHost);
  auto host_x = [&]() {
    auto x = host_x0;
    for (auto i = 0; i < n_var; ++i) {
      for (auto j = 0; j < BTHE21::n_stages; ++j) {
        x[i] += b[j] * host_ks[j * n_var + i] * dt;
      }
    }
    return x;
  }();
  auto host_error_estimate = [&]() {
    auto error_estimate = std::vector<double>(n_var);
    for (auto i = 0; i < n_var; ++i) {
      error_estimate[i] = 0.0;
      for (auto j = 0; j < BTHE21::n_stages; ++j) {
        error_estimate[i] += db[j] * host_ks[j * n_var + i] * dt;
      }
    }
    return error_estimate;
  }();
  for (auto i = 0; i < n_var; ++i) {
    EXPECT_DOUBLE_EQ(host_x[i], host_cuda_x[i]);
    EXPECT_DOUBLE_EQ(host_error_estimate[i], host_cuda_error_estimate[i]);
  }

  cudaFree(dev_error_estimate);
  cudaFree(dev_x);
  cudaFree(dev_ks);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, UpdateStateAndErrorLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto host_ks = std::vector<double>(n_var * BTHE21::n_stages);
  std::iota(host_ks.begin(), host_ks.end(), 1.0);
  auto dt = 0.1;
  auto b = BTHE21::b;
  auto db = []() {
    auto db = BTHE21::b;
    for (auto i = 0; i < BTHE21::n_stages; ++i) {
      db[i] -= BTHE21::bt[i];
    }
    return db;
  }();
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_ks = nullptr;
  cudaMalloc(&dev_ks, n_var * BTHE21::n_stages * sizeof(double));
  cudaMemcpy(dev_ks, host_ks.data(), n_var * BTHE21::n_stages * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  double* dev_error_estimate = nullptr;
  cudaMalloc(&dev_error_estimate, n_var * sizeof(double));

  cuda_update_state_and_error<n_var, BTHE21>
      <<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_ks, dt, dev_x,
                                            dev_error_estimate, b, db,
                                            BTHE21::n_stages);

  auto host_cuda_x = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_x.data(), dev_x, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  auto host_cuda_error_estimate = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_error_estimate.data(), dev_error_estimate,
             n_var * sizeof(double), cudaMemcpyDeviceToHost);
  auto host_x = [&]() {
    auto x = host_x0;
    for (auto i = 0; i < n_var; ++i) {
      for (auto j = 0; j < BTHE21::n_stages; ++j) {
        x[i] += b[j] * host_ks[j * n_var + i] * dt;
      }
    }
    return x;
  }();
  auto host_error_estimate = [&]() {
    auto error_estimate = std::vector<double>(n_var);
    for (auto i = 0; i < n_var; ++i) {
      error_estimate[i] = 0.0;
      for (auto j = 0; j < BTHE21::n_stages; ++j) {
        error_estimate[i] += db[j] * host_ks[j * n_var + i] * dt;
      }
    }
    return error_estimate;
  }();
  for (auto i = 0; i < n_var; ++i) {
    EXPECT_DOUBLE_EQ(host_x[i], host_cuda_x[i]);
    EXPECT_DOUBLE_EQ(host_error_estimate[i], host_cuda_error_estimate[i]);
  }

  cudaFree(dev_error_estimate);
  cudaFree(dev_x);
  cudaFree(dev_ks);
  cudaFree(dev_x0);
}
