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
                              double const* dt, int stage, int n_var,
                              aRow a_row) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    temp_state[i] = x0[i];
    for (auto j = 0; j < stage; ++j) {
      temp_state[i] += a_row[j] * ks[j * n_var + i] * *dt;
    }
    i += blockDim.x * gridDim.x;
  }
}

template <int n_var, typename RKMethod, typename ODE>
void cuda_evaluate_stages(double const* dev_x0, double* dev_temp_state,
                          double* dev_ks, double const* dev_dt) {
  ODE::compute_rhs(dev_x0, dev_ks);
  for (auto stage = 1; stage < RKMethod::n_stages; ++stage) {
    cudaMemset(dev_temp_state, 0, n_var * sizeof(double));
    cuda_rk_stage<<<num_blocks<n_var>(), block_size>>>(
        dev_x0, dev_ks, dev_temp_state, dev_dt, stage, n_var,
        RKMethod::a[stage - 1]);
    ODE::compute_rhs(dev_temp_state, dev_ks + stage * n_var);
  }
}

template <int n_var, typename RKMethod, typename bArray>
__global__ void cuda_update_state_and_error(double const* x0, double const* ks,
                                            double const* dt, double* x,
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
    x[i] *= *dt;
    x[i] += x0[i];
    i += blockDim.x * gridDim.x;
  }
}

template <int n_var, typename RKMethod, typename ODE, typename Output>
void cuda_integrate(double* dev_x0, double* dev_t0, double* dev_tf,
                    double* dev_atol, double* dev_rtol, Output& output) {
  auto constexpr db = []() {
    auto db = RKMethod::b;
    for (auto i = 0; i < RKMethod::b.size(); ++i) {
      db[i] -= RKMethod::bt[i];
    }
    return db;
  }();
  double* dev_ks = nullptr;
  cudaMalloc(&dev_ks, n_var * RKMethod::n_stages * sizeof(double));
  double* dev_dt = nullptr;
  cudaMalloc(&dev_dt, sizeof(double));
  cuda_estimate_initial_step<n_var, ODE>(dev_x0, dev_atol, dev_rtol, dev_dt);

  double* dev_t = nullptr;
  cudaMalloc(&dev_t, sizeof(double));
  cudaMemcpy(dev_t, dev_t0, sizeof(double), cudaMemcpyDeviceToDevice);
  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  cudaMemcpy(dev_x, dev_x0, n_var * sizeof(double), cudaMemcpyDeviceToDevice);
  double* dev_temp_state = nullptr;
  cudaMalloc(&dev_temp_state, n_var * sizeof(double));
  double* dev_error_estimate = nullptr;
  cudaMalloc(&dev_error_estimate, n_var * sizeof(double));
  double* dev_error_target = nullptr;
  cudaMalloc(&dev_error_target, n_var * sizeof(double));

  auto t = 0.0;
  cudaMemcpy(&t, dev_t, sizeof(double), cudaMemcpyDeviceToHost);
  auto tf = 0.0;
  cudaMemcpy(&tf, dev_tf, sizeof(double), cudaMemcpyDeviceToHost);
  output.save_state(dev_t, dev_x);
  while (t < tf) {
    cuda_evaluate_stages<n_var, RKMethod, ODE>(dev_x0, dev_temp_state, dev_ks,
                                               dev_dt);

    cuda_update_state_and_error<n_var, RKMethod>
        <<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_ks, dev_dt, dev_x,
                                              dev_error_estimate, RKMethod::b,
                                              db, RKMethod::n_stages);

    // TODO

    cudaMemcpy(&t, dev_t, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tf, dev_tf, sizeof(double), cudaMemcpyDeviceToHost);
  }

  cudaFree(dev_error_target);
  cudaFree(dev_error_estimate);
  cudaFree(dev_temp_state);
  cudaFree(dev_x);
  cudaFree(dev_t);
  cudaFree(dev_dt);
  cudaFree(dev_ks);
}
