#pragma once

#include <memory>
#include <span>

template <typename UnaryOp>
__global__ void elementwise_unary_op_kernel(double* v, int n,
                                            UnaryOp unary_op) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    unary_op(v[i]);
    i += blockDim.x * gridDim.x;
  }
}

template <typename UnaryOp>
__global__ void elementwise_unary_op_kernel(double const* u, double* v, int n,
                                            UnaryOp unary_op) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    v[i] = unary_op(u[i]);
    i += blockDim.x * gridDim.x;
  }
}

template <typename BinaryOp>
__global__ void elementwise_binary_op_kernel(double const* u, double* v, int n,
                                             BinaryOp binary_op) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    binary_op(u[i], v[i]);
    i += blockDim.x * gridDim.x;
  }
}

template <typename BinaryOp>
__global__ void elementwise_binary_op_kernel(double const* u, double const* v,
                                             double* w, int n,
                                             BinaryOp binary_op) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    w[i] = binary_op(u[i], v[i]);
    i += blockDim.x * gridDim.x;
  }
}

__global__ void inner_product_kernel(double const* u, double const* v,
                                     double* result, int n) {
  __shared__ double cache[256];  // TODO: magic number
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto partial_sum = 0.0;
  while (i < n) {
    partial_sum += u[i] * v[i];
    i += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = partial_sum;

  __syncthreads();
  for (auto reduction_width = blockDim.x / 2; reduction_width > 0;
       reduction_width /= 2) {
    if (threadIdx.x < reduction_width) {
      cache[threadIdx.x] += cache[threadIdx.x + reduction_width];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(result, cache[0]);
  }
}

template <int N>
class CudaState {
 public:
  CudaState() { cudaMalloc(&dev_state_, N * sizeof(double)); }
  CudaState(std::span<double const, N> state) {
    cudaMalloc(&dev_state_, N * sizeof(double));
    cudaMemcpy(dev_state_, state.data(), N * sizeof(double),
               cudaMemcpyHostToDevice);
  }
  CudaState(CudaState const& v) {
    cudaMalloc(&dev_state_, N * sizeof(double));
    cudaMemcpy(dev_state_, v.dev_state_, N * sizeof(double),
               cudaMemcpyDeviceToDevice);
  }
  CudaState(CudaState&& v) noexcept { std::swap(dev_state_, v.dev_state_); }
  auto& operator=(CudaState const& v) {
    if (this != &v) {
      cudaMemcpy(dev_state_, v.dev_state_, N * sizeof(double),
                 cudaMemcpyDeviceToDevice);
    }
    return *this;
  }
  auto& operator=(CudaState&& v) noexcept {
    std::swap(dev_state_, v.dev_state_);
    return *this;
  }
  ~CudaState() { cudaFree(dev_state_); }

  void to_host(std::span<double, N> state) const {
    cudaMemcpy(state.data(), dev_state_, N * sizeof(double),
               cudaMemcpyDeviceToHost);
  }

  auto static constexpr size() { return N; }

  operator double*() { return dev_state_; }
  operator double const*() const { return dev_state_; }

  template <typename UnaryOp>
  friend void elementwise_unary_op(CudaState& v, UnaryOp unary_op) {
    elementwise_unary_op_kernel<<<num_blocks, block_size>>>(v.dev_state_, N,
                                                            unary_op);
  }

  template <typename UnaryOp>
  friend void elementwise_unary_op(CudaState const& u, CudaState& v,
                                   UnaryOp unary_op) {
    elementwise_unary_op_kernel<<<num_blocks, block_size>>>(
        u.dev_state_, v.dev_state_, N, unary_op);
  }

  template <typename BinaryOp>
  friend void elementwise_binary_op(CudaState const& u, CudaState& v,
                                    BinaryOp binary_op) {
    elementwise_binary_op_kernel<<<num_blocks, block_size>>>(
        u.dev_state_, v.dev_state_, N, binary_op);
  }

  template <typename BinaryOp>
  friend void elementwise_binary_op(CudaState const& u, CudaState const& v,
                                    CudaState& w, BinaryOp binary_op) {
    elementwise_binary_op_kernel<<<num_blocks, block_size>>>(
        u.dev_state_, v.dev_state_, w.dev_state_, N, binary_op);
  }

  friend auto inner_product(CudaState const& u, CudaState const& v) {
    auto dev_result = static_cast<double*>(nullptr);
    cudaMalloc(&dev_result, sizeof(double));
    cudaMemset(dev_result, 0, sizeof(double));
    inner_product_kernel<<<num_blocks, block_size>>>(u.dev_state_, v.dev_state_,
                                                     dev_result, N);
    auto deleter = [](double* p) { cudaFree(p); };
    return std::unique_ptr<double, decltype(deleter)>(dev_result, deleter);
  }

 private:
  double* dev_state_{};

  auto static constexpr block_size = 256;
  auto static constexpr num_blocks = (N + block_size - 1) / block_size;
};
