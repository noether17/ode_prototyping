#pragma once

template <int n_var>
struct CUDAExpODE {
  static void compute_rhs(double const* x, double* f) {
    cudaMemcpy(f, x, n_var * sizeof(double), cudaMemcpyDeviceToDevice);
  }
};