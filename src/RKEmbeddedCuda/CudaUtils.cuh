#pragma once

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
