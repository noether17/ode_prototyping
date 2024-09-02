#pragma once

#include <iostream>
#include <vector>

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

template <int n_var>
struct RawCudaOutputWithProgress {
  std::vector<double> times{};
  std::vector<std::vector<double>> states{};

  void save_state(double t, double const* x_ptr) {
    auto host_x = std::vector<double>(n_var);
    cudaMemcpy(host_x.data(), x_ptr, n_var * sizeof(double),
               cudaMemcpyDeviceToHost);
    times.push_back(t);
    states.push_back(host_x);
    std::cout << "\rt = " << t;
  }

  ~RawCudaOutputWithProgress() { std::cout << '\n'; }
};
