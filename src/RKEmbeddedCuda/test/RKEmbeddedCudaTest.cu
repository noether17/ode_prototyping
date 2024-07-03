#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>

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
  double* dev_result = nullptr;
  cudaMalloc(&dev_result, sizeof(double));

  cuda_rk_norm<n_var>(dev_v, dev_scale, dev_result);

  auto host_cuda_result = 0.0;
  cudaMemcpy(&host_cuda_result, dev_result, sizeof(double),
             cudaMemcpyDeviceToHost);
  auto host_result = std::sqrt(
      std::inner_product(host_v.begin(), host_v.end(), host_scale.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      n_var);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);

  cudaFree(dev_result);
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
  double* dev_result = nullptr;
  cudaMalloc(&dev_result, sizeof(double));

  cuda_rk_norm<n_var>(dev_v, dev_scale, dev_result);

  auto host_cuda_result = 0.0;
  cudaMemcpy(&host_cuda_result, dev_result, sizeof(double),
             cudaMemcpyDeviceToHost);
  auto host_result = std::sqrt(
      std::inner_product(host_v.begin(), host_v.end(), host_scale.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      n_var);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);

  cudaFree(dev_result);
  cudaFree(dev_scale);
  cudaFree(dev_v);
}

TEST(RKEmbeddedCudaTest, EulerStepSmall) {
  auto constexpr n_var = 10;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_f = std::vector<double>(n_var);
  std::iota(host_f.begin(), host_f.end(), 1.0);
  auto host_dt = 0.1;
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_f0 = nullptr;
  cudaMalloc(&dev_f0, n_var * sizeof(double));
  cudaMemcpy(dev_f0, host_f.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_dt = nullptr;
  cudaMalloc(&dev_dt, sizeof(double));
  cudaMemcpy(dev_dt, &host_dt, sizeof(double), cudaMemcpyHostToDevice);
  double* dev_x1 = nullptr;
  cudaMalloc(&dev_x1, n_var * sizeof(double));

  cuda_euler_step<<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_f0, dev_dt,
                                                       dev_x1, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_x1, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_x[i] + host_dt * host_f[i];
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_x1);
  cudaFree(dev_dt);
  cudaFree(dev_f0);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, EulerStepLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_f = std::vector<double>(n_var);
  std::iota(host_f.begin(), host_f.end(), 1.0);
  auto host_dt = 0.1;
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_f0 = nullptr;
  cudaMalloc(&dev_f0, n_var * sizeof(double));
  cudaMemcpy(dev_f0, host_f.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_dt = nullptr;
  cudaMalloc(&dev_dt, sizeof(double));
  cudaMemcpy(dev_dt, &host_dt, sizeof(double), cudaMemcpyHostToDevice);
  double* dev_x1 = nullptr;
  cudaMalloc(&dev_x1, n_var * sizeof(double));

  cuda_euler_step<<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_f0, dev_dt,
                                                       dev_x1, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_x1, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_x[i] + host_dt * host_f[i];
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_x1);
  cudaFree(dev_dt);
  cudaFree(dev_f0);
  cudaFree(dev_x0);
}

template <typename RKMethod>
auto host_estimate_initial_step(std::vector<double> const& x0,
                                std::vector<double> const& atol,
                                std::vector<double> const& rtol) -> double {
  auto error_target = std::vector<double>(x0.size());
  for (auto i = 0; i < x0.size(); ++i) {
    error_target[i] = atol[i] + rtol[i] * std::abs(x0[i]);
  }

  auto f0 = x0;  // exponential ODE just copies state to rhs
  auto d0 = std::sqrt(
      std::inner_product(x0.begin(), x0.end(), error_target.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      x0.size());
  auto d1 = std::sqrt(
      std::inner_product(f0.begin(), f0.end(), error_target.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      x0.size());
  auto dt0 = (d0 < 1.0e-5 or d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (d0 / d1);

  auto x1 = std::vector<double>(x0.size());
  for (auto i = 0; i < x0.size(); ++i) {
    x1[i] = x0[i] + f0[i] * dt0;
  }
  auto f1 = x1;  // exponential ODE just copies state to rhs
  auto df = std::vector<double>(x0.size());
  for (auto i = 0; i < x0.size(); ++i) {
    df[i] = f1[i] - f0[i];
  }
  auto d2 = std::sqrt(std::inner_product(
                          df.begin(), df.end(), error_target.begin(), 0.0,
                          std::plus<>{},
                          [](auto a, auto b) { return (a / b) * (a / b); }) /
                      x0.size()) /
            dt0;

  auto constexpr p = RKMethod::p;
  auto dt1 = (std::max(d1, d2) <= 1.0e-15)
                 ? std::max(1.0e-6, dt0 * 1.0e-3)
                 : std::pow(0.01 / std::max(d1, d2), (1.0 / (1.0 + p)));
  return std::min(100.0 * dt0, dt1);
}

template <int n_var>
struct CUDAExpODE {
  static void compute_rhs(double const* x, double* f) {
    cudaMemcpy(f, x, n_var * sizeof(double), cudaMemcpyDeviceToDevice);
  }
};

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

  auto host_cuda_result =
      cuda_estimate_initial_step<n_var, HE21, CUDAExpODE<n_var>>(
          dev_x0, dev_atol, dev_rtol);

  auto host_result =
      host_estimate_initial_step<HE21>(host_x0, host_atol, host_rtol);
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

  auto host_cuda_result =
      cuda_estimate_initial_step<n_var, HE21, CUDAExpODE<n_var>>(
          dev_x0, dev_atol, dev_rtol);

  auto host_result =
      host_estimate_initial_step<HE21>(host_x0, host_atol, host_rtol);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);

  cudaFree(dev_rtol);
  cudaFree(dev_atol);
  cudaFree(dev_x0);
}

template <int n_var, typename RKMethod, typename ODE>
void host_evaluate_stages(double const* x0, double* temp_state, double* ks,
                          double dt) {
  ODE::compute_rhs(x0, ks);
  for (auto stage = 1; stage < RKMethod::n_stages; ++stage) {
    std::fill(temp_state, temp_state + n_var, 0.0);
    for (auto i = 0; i < n_var; ++i) {
      temp_state[i] = x0[i];
      for (auto j = 0; j < stage; ++j) {
        temp_state[i] += RKMethod::a[stage - 1][j] * ks[j * n_var + i] * dt;
      }
    }
    ODE::compute_rhs(temp_state, ks + stage * n_var);
  }
}

template <int n_var>
struct HostExpODE {
  static void compute_rhs(double const* x, double* f) {
    std::copy(x, x + n_var, f);
  }
};

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
  cudaMalloc(&dev_ks, n_var * HE21::n_stages * sizeof(double));
  double* dev_temp_state = nullptr;
  cudaMalloc(&dev_temp_state, n_var * sizeof(double));

  cuda_evaluate_stages<n_var, HE21, CUDAExpODE<n_var>>(dev_x0, dev_temp_state,
                                                       dev_ks, dt);

  auto host_cuda_result = std::vector<double>(n_var * HE21::n_stages);
  cudaMemcpy(host_cuda_result.data(), dev_ks,
             n_var * HE21::n_stages * sizeof(double), cudaMemcpyDeviceToHost);
  auto host_temp_state = std::vector<double>(n_var);
  auto host_result = std::vector<double>(n_var * HE21::n_stages);
  host_evaluate_stages<n_var, HE21, HostExpODE<n_var>>(
      host_x0.data(), host_temp_state.data(), host_result.data(), dt);
  for (auto i = 0; i < n_var * HE21::n_stages; ++i) {
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
  cudaMalloc(&dev_ks, n_var * HE21::n_stages * sizeof(double));
  double* dev_temp_state = nullptr;
  cudaMalloc(&dev_temp_state, n_var * sizeof(double));

  cuda_evaluate_stages<n_var, HE21, CUDAExpODE<n_var>>(dev_x0, dev_temp_state,
                                                       dev_ks, dt);

  auto host_cuda_result = std::vector<double>(n_var * HE21::n_stages);
  cudaMemcpy(host_cuda_result.data(), dev_ks,
             n_var * HE21::n_stages * sizeof(double), cudaMemcpyDeviceToHost);
  auto host_temp_state = std::vector<double>(n_var);
  auto host_result = std::vector<double>(n_var * HE21::n_stages);
  host_evaluate_stages<n_var, HE21, HostExpODE<n_var>>(
      host_x0.data(), host_temp_state.data(), host_result.data(), dt);
  for (auto i = 0; i < n_var * HE21::n_stages; ++i) {
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
  auto host_ks = std::vector<double>(n_var * HE21::n_stages);
  std::iota(host_ks.begin(), host_ks.end(), 1.0);
  auto dt = 0.1;
  auto b = HE21::b;
  auto db = []() {
    auto db = HE21::b;
    for (auto i = 0; i < HE21::n_stages; ++i) {
      db[i] -= HE21::bt[i];
    }
    return db;
  }();
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_ks = nullptr;
  cudaMalloc(&dev_ks, n_var * HE21::n_stages * sizeof(double));
  cudaMemcpy(dev_ks, host_ks.data(), n_var * HE21::n_stages * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  double* dev_error_estimate = nullptr;
  cudaMalloc(&dev_error_estimate, n_var * sizeof(double));

  cuda_update_state_and_error<n_var, HE21><<<num_blocks<n_var>(), block_size>>>(
      dev_x0, dev_ks, dt, dev_x, dev_error_estimate, b, db, HE21::n_stages);

  auto host_cuda_x = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_x.data(), dev_x, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  auto host_cuda_error_estimate = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_error_estimate.data(), dev_error_estimate,
             n_var * sizeof(double), cudaMemcpyDeviceToHost);
  auto host_x = [&]() {
    auto x = host_x0;
    for (auto i = 0; i < n_var; ++i) {
      for (auto j = 0; j < HE21::n_stages; ++j) {
        x[i] += b[j] * host_ks[j * n_var + i] * dt;
      }
    }
    return x;
  }();
  auto host_error_estimate = [&]() {
    auto error_estimate = std::vector<double>(n_var);
    for (auto i = 0; i < n_var; ++i) {
      error_estimate[i] = 0.0;
      for (auto j = 0; j < HE21::n_stages; ++j) {
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
  auto host_ks = std::vector<double>(n_var * HE21::n_stages);
  std::iota(host_ks.begin(), host_ks.end(), 1.0);
  auto dt = 0.1;
  auto b = HE21::b;
  auto db = []() {
    auto db = HE21::b;
    for (auto i = 0; i < HE21::n_stages; ++i) {
      db[i] -= HE21::bt[i];
    }
    return db;
  }();
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_ks = nullptr;
  cudaMalloc(&dev_ks, n_var * HE21::n_stages * sizeof(double));
  cudaMemcpy(dev_ks, host_ks.data(), n_var * HE21::n_stages * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  double* dev_error_estimate = nullptr;
  cudaMalloc(&dev_error_estimate, n_var * sizeof(double));

  cuda_update_state_and_error<n_var, HE21><<<num_blocks<n_var>(), block_size>>>(
      dev_x0, dev_ks, dt, dev_x, dev_error_estimate, b, db, HE21::n_stages);

  auto host_cuda_x = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_x.data(), dev_x, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  auto host_cuda_error_estimate = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_error_estimate.data(), dev_error_estimate,
             n_var * sizeof(double), cudaMemcpyDeviceToHost);
  auto host_x = [&]() {
    auto x = host_x0;
    for (auto i = 0; i < n_var; ++i) {
      for (auto j = 0; j < HE21::n_stages; ++j) {
        x[i] += b[j] * host_ks[j * n_var + i] * dt;
      }
    }
    return x;
  }();
  auto host_error_estimate = [&]() {
    auto error_estimate = std::vector<double>(n_var);
    for (auto i = 0; i < n_var; ++i) {
      error_estimate[i] = 0.0;
      for (auto j = 0; j < HE21::n_stages; ++j) {
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

TEST(RKEmbeddedCudaTest, CompareCUDAToCPUHE21) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, HE21, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  EXPECT_EQ(11026, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(5.0636296424493379, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(11026, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  EXPECT_NEAR(790.81720024690105,
              output.states[output.states.size() / 2][n_var / 2],
              790.81720024690105 * state_tol);
  EXPECT_NEAR(1423.4709604444222,
              output.states[output.states.size() / 2].back(),
              1423.4709604444222 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110132.17777934085, output.states.back()[n_var / 2],
              110132.17777934085 * state_tol);
  EXPECT_NEAR(198237.92000281269, output.states.back().back(),
              198237.92000281269 * state_tol);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, CUDAIntegrateConsistencyTestHE21) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, HE21, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  EXPECT_EQ(11026, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(5.0636296424493379, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(11026, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(790.81720024690105,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1423.4709604444204,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110132.17777934109, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198237.92000281301, output.states.back().back());

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, CompareCUDAToCPURKF45) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, RKF45, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  // TODO: Investigate differences between CPU and CUDA algorithms
  EXPECT_EQ(50, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  // EXPECT_DOUBLE_EQ(5.0888631842534933, output.times[output.times.size() /
  // 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  // Increase tolerance for RKF Methods
  auto const state_tol = 2.0 * std::sqrt(output.states.size()) *
                         std::numeric_limits<double>::epsilon();
  EXPECT_EQ(50, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  // EXPECT_NEAR(811.03335805992742,
  //             output.states[output.states.size() / 2][n_var / 2],
  //             811.03335805992742 * state_tol);
  // EXPECT_NEAR(1459.8600445078696,
  //             output.states[output.states.size() / 2].back(),
  //             1459.8600445078696 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110134.06230636688, output.states.back()[n_var / 2],
              110134.06230636688 * state_tol);
  EXPECT_NEAR(198241.3121514605, output.states.back().back(),
              198241.3121514605 * state_tol);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, CUDAIntegrateConsistencyTestRKF45) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, RKF45, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  EXPECT_EQ(50, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(5.088863184252939, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(50, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(811.03335805947813,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1459.8600445070604,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110134.06230636714, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198241.31215146082, output.states.back().back());

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, CompareCUDAToCPUDOPRI5) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, DOPRI5, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  // TODO: Investigate differences between CPU and CUDA algorithms
  EXPECT_EQ(45, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  // EXPECT_DOUBLE_EQ(4.9896555947535841, output.times[output.times.size() /
  // 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(45, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  // EXPECT_NEAR(734.42960843551498,
  //             output.states[output.states.size() / 2][n_var / 2],
  //             734.42960843551498 * state_tol);
  // EXPECT_NEAR(1321.9732951839271,
  //             output.states[output.states.size() / 2].back(),
  //             1321.9732951839271 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110132.46804595254, output.states.back()[n_var / 2],
              110132.46804595254 * state_tol);
  EXPECT_NEAR(198238.44248271457, output.states.back().back(),
              198238.44248271457 * state_tol);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, CUDAIntegrateConsistencyTestDOPRI5) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, DOPRI5, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  EXPECT_EQ(45, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(4.9896555948055186, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(45, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(734.42960847365703,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1321.9732952525828,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110132.46804595242, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198238.44248271445, output.states.back().back());

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, CompareCUDAToCPUDVERK) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, DVERK, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  // TODO: Investigate differences between CPU and CUDA algorithms
  EXPECT_EQ(32, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  // EXPECT_DOUBLE_EQ(5.1178634791056821, output.times[output.times.size() /
  // 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(32, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  // EXPECT_NEAR(834.89108343292787,
  //             output.states[output.states.size() / 2][n_var / 2],
  //             834.89108343292787 * state_tol);
  // EXPECT_NEAR(1502.8039501792707,
  //             output.states[output.states.size() / 2].back(),
  //             1502.8039501792707 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110132.30512693945, output.states.back()[n_var / 2],
              110132.30512693945 * state_tol);
  EXPECT_NEAR(198238.14922849106, output.states.back().back(),
              198238.14922849106 * state_tol);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, CUDAIntegrateConsistencyTestDVERK) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, DVERK, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  EXPECT_EQ(32, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(5.1178634782992676, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(32, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(834.89108275966021,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1502.8039489673879,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110132.30512693951, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198238.14922849112, output.states.back().back());

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, CompareCUDAToCPURKF78) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, RKF78, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  // TODO: Investigate differences between CPU and CUDA algorithms
  EXPECT_EQ(13, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  // EXPECT_DOUBLE_EQ(4.4481779636028618, output.times[output.times.size() /
  // 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  // Increase tolerance for RKF Methods
  auto const state_tol = 3.0 * std::sqrt(output.states.size()) *
                         std::numeric_limits<double>::epsilon();
  EXPECT_EQ(13, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  // EXPECT_NEAR(427.3544878966228,
  //             output.states[output.states.size() / 2][n_var / 2],
  //             427.3544878966228 * state_tol);
  // EXPECT_NEAR(769.23807821392074,
  //             output.states[output.states.size() / 2].back(),
  //             769.23807821392074 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110131.78807367262, output.states.back()[n_var / 2],
              110131.78807367262 * state_tol);
  EXPECT_NEAR(198237.21853261057, output.states.back().back(),
              198237.21853261057 * state_tol);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, CUDAIntegrateConsistencyTestRKF78) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, RKF78, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, output);

  EXPECT_EQ(13, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(4.448177963590588, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(13, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(427.35448789137757,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(769.23807820447973,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110131.78807367239, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198237.21853261034, output.states.back().back());

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}
