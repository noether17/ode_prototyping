#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "BTDOPRI5.hpp"
#include "BTDVERK.hpp"
#include "BTHE21.hpp"
#include "BTRKF45.hpp"
#include "BTRKF78.hpp"
#include "CudaNBodyOde.cuh"
#include "RKEmbeddedCuda.cuh"
#include "RawCudaOutput.cuh"

auto constexpr N = 1024;
auto constexpr L = 1.0;
auto constexpr n_var = N * 6;
auto init_state_and_tol() -> std::pair<double*, double*>;

template <typename Output>
void write_to_file(Output const& output, std::string const& filename) {
  auto output_file = std::ofstream(filename, std::ios::out | std::ios::binary);
  auto n_rows = output.times.size();
  auto n_cols = static_cast<std::size_t>(n_var + 1);
  output_file.write(reinterpret_cast<char const*>(&n_rows),
                    sizeof(std::size_t));
  output_file.write(reinterpret_cast<char const*>(&n_cols),
                    sizeof(std::size_t));
  for (auto i = 0; i < output.times.size(); ++i) {
    output_file.write(reinterpret_cast<char const*>(&output.times[i]),
                      sizeof(double));
    for (auto j = 0; j < n_var; ++j) {
      output_file.write(reinterpret_cast<char const*>(&output.states[i][j]),
                        sizeof(double));
    }
  }
}

template <typename ButcherTableau>
auto bt_to_string() -> std::string;

template <>
auto bt_to_string<BTHE21>() -> std::string {
  return "HE21";
}

template <>
auto bt_to_string<BTRKF45>() -> std::string {
  return "RKF45";
}

template <>
auto bt_to_string<BTDOPRI5>() -> std::string {
  return "DOPRI5";
}

template <>
auto bt_to_string<BTDVERK>() -> std::string {
  return "DVERK";
}

template <>
auto bt_to_string<BTRKF78>() -> std::string {
  return "RKF78";
}

template <typename ButcherTableau>
auto run_simulation() {
  auto bt_name = bt_to_string<ButcherTableau>();
  std::cout << "Starting simulation using " << bt_name << " method.\n";

  auto t0 = 0.0;
  auto tf = std::sqrt(L * L * L / N);
  std::cout << "End time = " << tf << '\n';

  auto [dev_x0, dev_tol] = init_state_and_tol();
  auto ode = CudaNBodyOde<n_var>{1.0e-3};
  auto output = RawCudaOutputWithProgress<n_var>{};

  auto start = std::chrono::steady_clock::now();
  cuda_integrate<n_var, ButcherTableau, CudaNBodyOde<n_var>,
                 RawCudaOutputWithProgress<n_var>>(dev_x0, t0, tf, dev_tol,
                                                   dev_tol, ode, output);
  auto duration = std::chrono::steady_clock::now() - start;

  auto filename =
      "cuda_n_body_" + std::to_string(N) + "_particles_" + bt_name + ".bin";
  write_to_file(output, filename);

  cudaFree(dev_tol);
  cudaFree(dev_x0);

  return duration.count();
}

int main() {
  // auto duration = run_simulation<BTHE21>() * 1.0e-9;
  // std::cout << "Completed in " << duration << "s.\n";

  auto duration = run_simulation<BTRKF45>() * 1.0e-9;
  std::cout << "Completed in " << duration << "s.\n";

  duration = run_simulation<BTDOPRI5>() * 1.0e-9;
  std::cout << "Completed in " << duration << "s.\n";

  duration = run_simulation<BTDVERK>() * 1.0e-9;
  std::cout << "Completed in " << duration << "s.\n";

  duration = run_simulation<BTRKF78>() * 1.0e-9;
  std::cout << "Completed in " << duration << "s.\n";
}

auto init_state_and_tol() -> std::pair<double*, double*> {
  auto host_x0 = std::vector<double>(n_var);
  auto gen = std::mt19937{0};
  auto dist = std::uniform_real_distribution<double>(0.0, L);
  for (auto i = 0; i < host_x0.size() / 2; ++i) {
    host_x0[i] = dist(gen);
  }
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-10);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  return {dev_x0, dev_tol};
}
