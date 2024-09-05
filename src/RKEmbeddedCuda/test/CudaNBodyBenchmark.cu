#include <benchmark/benchmark.h>

#include <numeric>

#include "CudaNBodyOde.cuh"

#define REPEAT2(X) X X
#define REPEAT4(X) REPEAT2(X) REPEAT2(X)
#define REPEAT(X) REPEAT4(X) REPEAT4(X)

auto constexpr n_repetitions = 16;

static void BM_NBodySimple(benchmark::State& state) {
  auto constexpr n_particles = 1024;
  auto constexpr n_var = n_particles * 6;
  auto host_x = []() {
    auto x = std::array<double, n_var>{};
    std::iota(x.begin(), x.end(), 0.0);
    return x;
  }();

  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  cudaMemcpy(dev_x, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_f = nullptr;
  cudaMalloc(&dev_f, n_var * sizeof(double));

  auto simple_n_body = CudaNBodyOdeSimple<n_var>{};
  for (auto _ : state) {
    REPEAT(benchmark::DoNotOptimize(dev_f);
           simple_n_body.compute_rhs(dev_x, dev_f); benchmark::ClobberMemory();)
  }

  cudaMemcpy(host_x.data(), dev_f, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  benchmark::DoNotOptimize(host_x.data());
  benchmark::ClobberMemory();

  state.SetItemsProcessed(state.iterations() * n_particles * n_repetitions);

  cudaFree(dev_f);
  cudaFree(dev_x);
}

BENCHMARK(BM_NBodySimple);

static void BM_NBodyPairwise(benchmark::State& state) {
  auto constexpr n_particles = 1024;
  auto constexpr n_var = n_particles * 6;
  auto host_x = []() {
    auto x = std::array<double, n_var>{};
    std::iota(x.begin(), x.end(), 0.0);
    return x;
  }();

  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  cudaMemcpy(dev_x, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_f = nullptr;
  cudaMalloc(&dev_f, n_var * sizeof(double));

  auto pairwise_n_body = CudaNBodyOde<n_var>{};
  for (auto _ : state) {
    REPEAT(benchmark::DoNotOptimize(dev_f);
           pairwise_n_body.compute_rhs(dev_x, dev_f);
           benchmark::ClobberMemory();)
  }

  cudaMemcpy(host_x.data(), dev_f, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  benchmark::DoNotOptimize(host_x.data());
  benchmark::ClobberMemory();

  state.SetItemsProcessed(state.iterations() * n_particles * n_repetitions);

  cudaFree(dev_f);
  cudaFree(dev_x);
}

BENCHMARK(BM_NBodyPairwise);
