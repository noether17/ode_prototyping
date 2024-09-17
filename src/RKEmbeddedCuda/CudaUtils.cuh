#pragma once

auto static constexpr block_size = 256;

template <int N>
auto consteval num_blocks() {
  return (N + block_size - 1) / block_size;
}

__global__ void elementwise_add(double* a, double* b, double* c, int n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    c[i] = a[i] + b[i];
    i += blockDim.x * gridDim.x;
  }
}

template <typename UnaryOp>
__global__ void elementwise_unary_op_kernel(double* v, int n,
                                            UnaryOp unary_op) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    v[i] = unary_op(v[i]);
    i += blockDim.x * gridDim.x;
  }
}
