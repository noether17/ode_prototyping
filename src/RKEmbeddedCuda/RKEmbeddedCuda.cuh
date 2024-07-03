#pragma once

#include <array>
#include <cmath>

auto static constexpr block_size = 256;

template <int N>
auto consteval num_blocks() {
  return (N + block_size - 1) / block_size;
}

struct HE21 {
  auto static constexpr a = std::array<std::array<double, 1>, 1>{{1.0}};
  auto static constexpr b = std::array{1.0 / 2.0, 1.0 / 2.0};
  auto static constexpr bt = std::array{1.0, 0.0};
  auto static constexpr p = 2;
  auto static constexpr pt = 1;
  auto static constexpr n_stages = static_cast<int>(b.size());
};

struct RKF45 {
  auto static constexpr a = std::array{
      std::array{1.0 / 4.0, 0.0, 0.0, 0.0, 0.0},
      std::array{3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0},
      std::array{1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0},
      std::array{439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0},
      std::array{-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0,
                 -11.0 / 40.0}};
  auto static constexpr b = std::array{25.0 / 216.0,    0.0,  1408.0 / 2565.0,
                                       2197.0 / 4104.0, -0.2, 0.0};
  auto static constexpr bt =
      std::array{16.0 / 135.0,      0.0,         6656.0 / 12825.0,
                 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0};
  auto static constexpr p = 4;
  auto static constexpr pt = 5;
  auto static constexpr n_stages = static_cast<int>(b.size());
};

struct DOPRI5 {
  auto static constexpr a = std::array{
      std::array{1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
      std::array{44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0},
      std::array{19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0,
                 -212.0 / 729.0, 0.0, 0.0},
      std::array{9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
                 -5103.0 / 18656.0, 0.0},
      std::array{35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
                 -2187.0 / 6784.0, 11.0 / 84.0}};
  auto static constexpr b = std::array{
      35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
      11.0 / 84.0,  0.0};
  auto static constexpr bt = std::array{5179.0 / 57600.0,    0.0,
                                        7571.0 / 16695.0,    393.0 / 640.0,
                                        -92097.0 / 339200.0, 187.0 / 2100.0,
                                        1.0 / 40.0};
  auto static constexpr p = 5;
  auto static constexpr pt = 4;
  auto static constexpr n_stages = static_cast<int>(b.size());
};

struct DVERK {
  auto static constexpr a = std::array{
      std::array{1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{4.0 / 75.0, 16.0 / 75.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{5.0 / 6.0, -8.0 / 3.0, 5.0 / 2.0, 0.0, 0.0, 0.0, 0.0},
      std::array{-165.0 / 64.0, 55.0 / 6.0, -425.0 / 64.0, 85.0 / 96.0, 0.0,
                 0.0, 0.0},
      std::array{12.0 / 5.0, -8.0, 4015.0 / 612.0, -11.0 / 36.0, 88.0 / 255.0,
                 0.0, 0.0},
      std::array{-8263.0 / 15000.0, 124.0 / 75.0, -643.0 / 680.0, -81.0 / 250.0,
                 2484.0 / 10625.0, 0.0, 0.0},
      std::array{3501.0 / 1720.0, -300.0 / 43.0, 297275.0 / 52632.0,
                 -319.0 / 2322.0, 24068.0 / 84065.0, 0.0, 3850.0 / 26703.0}};
  auto static constexpr b =
      std::array{3.0 / 40.0,     0.0, 875.0 / 2244.0,  23.0 / 72.0,
                 264.0 / 1955.0, 0.0, 125.0 / 11592.0, 43.0 / 616.0};
  auto static constexpr bt = std::array{
      13.0 / 160.0, 0.0, 2375.0 / 5984.0, 5.0 / 16.0, 12.0 / 85.0, 3.0 / 44.0,
      0.0,          0.0};
  auto static constexpr p = 6;
  auto static constexpr pt = 5;
  auto static constexpr n_stages = static_cast<int>(b.size());
};

struct RKF78 {
  auto static constexpr a = std::array{
      std::array{2.0 / 27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0},
      std::array{1.0 / 36.0, 1.0 / 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0},
      std::array{1.0 / 24.0, 0.0, 1.0 / 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0},
      std::array{5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0},
      std::array{1.0 / 20.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 5.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0},
      std::array{-25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -65.0 / 27.0,
                 125.0 / 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0,
                 13.0 / 900.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      std::array{2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0,
                 67.0 / 90.0, 3.0, 0.0, 0.0, 0.0, 0.0},
      std::array{-91.0 / 108.0, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0,
                 311.0 / 54.0, -19.0 / 60.0, 17.0 / 6.0, -1.0 / 12.0, 0.0, 0.0,
                 0.0},
      std::array{2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0,
                 -301.0 / 82.0, 2133.0 / 4100.0, 45.0 / 82.0, 45.0 / 164.0,
                 18.0 / 41.0, 0.0, 0.0},
      std::array{3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0,
                 -3.0 / 41.0, 3.0 / 41.0, 6.0 / 41.0, 0.0, 0.0},
      std::array{-1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0,
                 -289.0 / 82.0, 2193.0 / 4100.0, 51.0 / 82.0, 33.0 / 164.0,
                 12.0 / 41.0, 0.0, 1.0}};
  auto static constexpr b =
      std::array{41.0 / 840.0, 0.0,        0.0,        0.0,         0.0,
                 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0, 9.0 / 280.0, 9.0 / 280.0,
                 41.0 / 840.0, 0.0,        0.0};
  auto static constexpr bt = std::array{
      0.0,          0.0,          0.0,         0.0,         0.0,
      34.0 / 105.0, 9.0 / 35.0,   9.0 / 35.0,  9.0 / 280.0, 9.0 / 280.0,
      0.0,          41.0 / 840.0, 41.0 / 840.0};
  auto static constexpr p = 7;
  auto static constexpr pt = 8;
  auto static constexpr n_stages = static_cast<int>(b.size());
};

__global__ void cuda_compute_error_target(double const* x, double const* rtol,
                                          double const* atol,
                                          double* error_target, int n_var) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    error_target[i] = atol[i] + rtol[i] * std::abs(x[i]);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void cuda_rk_norm_reduction_step(double const* v,
                                            double const* scale, double* temp,
                                            int n_var) {
  __shared__ double cache[block_size];
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto cache_index = threadIdx.x;
  auto temp_index = blockIdx.x;
  auto thread_value = 0.0;
  while (i < n_var) {
    auto scaled_value = v[i] / scale[i];
    thread_value += scaled_value * scaled_value;
    i += blockDim.x * gridDim.x;
  }

  cache[cache_index] = thread_value;
  __syncthreads();
  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (cache_index < stride) {
      cache[cache_index] += cache[cache_index + stride];
    }
    __syncthreads();
  }

  if (cache_index == 0) {
    temp[temp_index] = cache[0];
  }
}

__global__ void cuda_rk_norm_reduction_final(double const* temp, double* result,
                                             int n_var, int temp_size) {
  __shared__ double cache[block_size];
  auto i = threadIdx.x;  // final reduction is always a single block
  auto thread_value = 0.0;
  while (i < temp_size) {
    thread_value += temp[i];
    i += blockDim.x;
  }

  cache[threadIdx.x] = thread_value;
  __syncthreads();
  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      cache[threadIdx.x] += cache[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *result = std::sqrt(cache[0] / n_var);
  }
}

template <int n_var>
void cuda_rk_norm(double const* dev_v, double const* dev_scale,
                  double* dev_result) {
  double* dev_temp = nullptr;
  cudaMalloc(&dev_temp, num_blocks<n_var>() * sizeof(double));

  cuda_rk_norm_reduction_step<<<num_blocks<n_var>(), block_size>>>(
      dev_v, dev_scale, dev_temp, n_var);
  cuda_rk_norm_reduction_final<<<1, block_size>>>(dev_temp, dev_result, n_var,
                                                  num_blocks<n_var>());
  cudaDeviceSynchronize();

  cudaFree(dev_temp);
}

__global__ void cuda_compute_dt0(double const* d0, double const* d1,
                                 double* dt0) {
  if (threadIdx.x == 0) {
    *dt0 = (*d0 < 1.0e-5 or *d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (*d0 / *d1);
  }
}

__global__ void cuda_euler_step(double const* x0, double const* f0,
                                double const* dt, double* x1, int n_var) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    x1[i] = x0[i] + *dt * f0[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void cuda_vector_diff(double const* x0, double const* x1, double* dx,
                                 int n_var) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    dx[i] = x1[i] - x0[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void cuda_divide(double const* a, double const* b, double* c,
                            int n_var) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    c[i] = a[i] / b[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__ void cuda_compute_dt(double const* d1, double const* d2,
                                double const* dt0, int p, double* dt) {
  if (threadIdx.x == 0) {
    auto dt1 = (std::max(*d1, *d2) <= 1.0e-15)
                   ? std::max(1.0e-6, *dt0 * 1.0e-3)
                   : std::pow(0.01 / std::max(*d1, *d2), (1.0 / (1.0 + p)));
    *dt = std::min(100.0 * *dt0, dt1);
  }
}

template <int n_var, typename RKMethod, typename ODE>
void cuda_estimate_initial_step(double* dev_x0, double* dev_atol,
                                double* dev_rtol, double* dev_dt) {
  double* dev_error_target = nullptr;
  cudaMalloc(&dev_error_target, n_var * sizeof(double));
  cuda_compute_error_target<<<num_blocks<n_var>(), block_size>>>(
      dev_x0, dev_rtol, dev_atol, dev_error_target, n_var);

  double* dev_f0 = nullptr;
  cudaMalloc(&dev_f0, n_var * sizeof(double));
  ODE::compute_rhs(dev_x0, dev_f0);
  double* dev_d0 = nullptr;
  cudaMalloc(&dev_d0, sizeof(double));
  cuda_rk_norm<n_var>(dev_x0, dev_error_target, dev_d0);
  double* dev_d1 = nullptr;
  cudaMalloc(&dev_d1, sizeof(double));
  cuda_rk_norm<n_var>(dev_f0, dev_error_target, dev_d1);
  double* dev_dt0 = nullptr;
  cudaMalloc(&dev_dt0, sizeof(double));
  cuda_compute_dt0<<<1, 1>>>(dev_d0, dev_d1, dev_dt0);

  double* dev_x1 = nullptr;
  cudaMalloc(&dev_x1, n_var * sizeof(double));
  cuda_euler_step<<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_f0, dev_dt0,
                                                       dev_x1, n_var);
  double* dev_f1 = nullptr;
  cudaMalloc(&dev_f1, n_var * sizeof(double));
  ODE::compute_rhs(dev_x1, dev_f1);
  double* dev_df = nullptr;
  cudaMalloc(&dev_df, n_var * sizeof(double));
  cuda_vector_diff<<<num_blocks<n_var>(), block_size>>>(dev_f0, dev_f1, dev_df,
                                                        n_var);
  double* dev_d2 = nullptr;
  cudaMalloc(&dev_d2, sizeof(double));
  cuda_rk_norm<n_var>(dev_df, dev_error_target, dev_d2);
  cuda_divide<<<1, 1>>>(dev_d2, dev_dt0, dev_d2, 1);

  cuda_compute_dt<<<1, 1>>>(dev_d1, dev_d2, dev_dt0, RKMethod::p, dev_dt);

  cudaFree(dev_d2);
  cudaFree(dev_df);
  cudaFree(dev_f1);
  cudaFree(dev_x1);
  cudaFree(dev_dt0);
  cudaFree(dev_d1);
  cudaFree(dev_d0);
  cudaFree(dev_f0);
  cudaFree(dev_error_target);
}

template <typename aRow>
__global__ void cuda_rk_stage(double const* x0, double* ks, double* temp_state,
                              double dt, int stage, int n_var, aRow a_row) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    temp_state[i] = x0[i];
    for (auto j = 0; j < stage; ++j) {
      temp_state[i] += a_row[j] * ks[j * n_var + i] * dt;
    }
    i += blockDim.x * gridDim.x;
  }
}

template <int n_var, typename RKMethod, typename ODE>
void cuda_evaluate_stages(double const* dev_x0, double* dev_temp_state,
                          double* dev_ks, double dt) {
  ODE::compute_rhs(dev_x0, dev_ks);
  for (auto stage = 1; stage < RKMethod::n_stages; ++stage) {
    cudaMemset(dev_temp_state, 0, n_var * sizeof(double));
    cuda_rk_stage<<<num_blocks<n_var>(), block_size>>>(
        dev_x0, dev_ks, dev_temp_state, dt, stage, n_var,
        RKMethod::a[stage - 1]);
    ODE::compute_rhs(dev_temp_state, dev_ks + stage * n_var);
  }
}

template <int n_var, typename RKMethod, typename bArray>
__global__ void cuda_update_state_and_error(double const* x0, double const* ks,
                                            double dt, double* x,
                                            double* error_estimate, bArray b,
                                            bArray db, int stages) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    x[i] = 0.0;
    error_estimate[i] = 0.0;
    for (auto j = 0; j < stages; ++j) {
      x[i] += b[j] * ks[j * n_var + i];
      error_estimate[i] += db[j] * ks[j * n_var + i];
    }
    x[i] *= dt;
    x[i] += x0[i];
    error_estimate[i] *= dt;
    i += blockDim.x * gridDim.x;
  }
}

__global__ void cuda_compute_error_target(double const* x1, double const* x2,
                                          double const* rtol,
                                          double const* atol,
                                          double* error_target, int n_var) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    error_target[i] =
        atol[i] + rtol[i] * std::max(std::abs(x1[i]), std::abs(x2[i]));
    i += blockDim.x * gridDim.x;
  }
}

__global__ void cuda_add(double const* a, double const* b, double* c,
                         int n_var) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    c[i] = a[i] + b[i];
    i += blockDim.x * gridDim.x;
  }
}

template <int n_var, typename RKMethod, typename ODE, typename Output>
void cuda_integrate(double* dev_x0, double t0, double tf, double* dev_atol,
                    double* dev_rtol, Output& output) {
  auto constexpr max_step_scale = 6.0;
  auto constexpr min_step_scale = 0.33;
  auto constexpr db = []() {
    auto db = RKMethod::b;
    for (auto i = 0; i < RKMethod::b.size(); ++i) {
      db[i] -= RKMethod::bt[i];
    }
    return db;
  }();
  auto constexpr q = std::min(RKMethod::p, RKMethod::pt);
  auto const safety_factor = std::pow(0.38, (1.0 / (1.0 + q)));
  double* dev_ks = nullptr;
  cudaMalloc(&dev_ks, n_var * RKMethod::n_stages * sizeof(double));
  double* dev_dt = nullptr;
  cudaMalloc(&dev_dt, sizeof(double));
  cuda_estimate_initial_step<n_var, RKMethod, ODE>(dev_x0, dev_atol, dev_rtol,
                                                   dev_dt);
  auto dt = 0.0;
  cudaMemcpy(&dt, dev_dt, sizeof(double), cudaMemcpyDeviceToHost);

  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  cudaMemcpy(dev_x, dev_x0, n_var * sizeof(double), cudaMemcpyDeviceToDevice);
  double* dev_temp_state = nullptr;
  cudaMalloc(&dev_temp_state, n_var * sizeof(double));
  double* dev_error_estimate = nullptr;
  cudaMalloc(&dev_error_estimate, n_var * sizeof(double));
  double* dev_error_target = nullptr;
  cudaMalloc(&dev_error_target, n_var * sizeof(double));
  double* dev_scaled_error = nullptr;
  cudaMalloc(&dev_scaled_error, sizeof(double));

  auto t = t0;
  output.save_state(t, dev_x);
  while (t < tf) {
    cuda_evaluate_stages<n_var, RKMethod, ODE>(dev_x0, dev_temp_state, dev_ks,
                                               dt);

    cuda_update_state_and_error<n_var, RKMethod>
        <<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_ks, dt, dev_x,
                                              dev_error_estimate, RKMethod::b,
                                              db, RKMethod::n_stages);

    cuda_compute_error_target<<<num_blocks<n_var>(), block_size>>>(
        dev_x0, dev_x, dev_rtol, dev_atol, dev_error_target, n_var);
    cuda_rk_norm<n_var>(dev_error_estimate, dev_error_target, dev_scaled_error);

    auto host_scaled_error = 0.0;
    cudaMemcpy(&host_scaled_error, dev_scaled_error, sizeof(double),
               cudaMemcpyDeviceToHost);
    if (host_scaled_error < 1.0) {
      t += dt;
      cudaMemcpy(dev_x0, dev_x, n_var * sizeof(double),
                 cudaMemcpyDeviceToDevice);
      output.save_state(t, dev_x);
    }

    auto dtnew =
        dt * safety_factor / std::pow(host_scaled_error, 1.0 / (1.0 + q));
    if (std::abs(dtnew) > max_step_scale * std::abs(dt)) {
      dt *= max_step_scale;
    } else if (std::abs(dtnew) < min_step_scale * std::abs(dt)) {
      dt *= min_step_scale;
    } else if (dtnew / (tf - t0) < 1.0e-12) {
      dt = (tf - t0) * 1.0e-12;
    } else {
      dt = dtnew;
    }
    if (t + dt > tf) {
      dt = tf - t;
    }
  }

  cudaFree(dev_scaled_error);
  cudaFree(dev_error_target);
  cudaFree(dev_error_estimate);
  cudaFree(dev_temp_state);
  cudaFree(dev_x);
  cudaFree(dev_dt);
  cudaFree(dev_ks);
}
