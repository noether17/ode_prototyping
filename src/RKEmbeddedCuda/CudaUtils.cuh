#pragma once

__global__ void elementwise_add(double* a, double* b, double* c, int n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    c[i] = a[i] + b[i];
    i += blockDim.x * gridDim.x;
  }
}
