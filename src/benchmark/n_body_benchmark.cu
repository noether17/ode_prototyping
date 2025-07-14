#include <benchmark/benchmark.h>

#include <concepts>

#include "BTDOPRI5.hpp"
#include "BTDVERK.hpp"
#include "BTHE21.hpp"
#include "BTRKF45.hpp"
#include "BTRKF78.hpp"
#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "ODEState.hpp"
#include "RKEmbeddedParallel.hpp"
#include "SingleThreadedExecutor.hpp"
#include "SpinningParticlesInBox.hpp"
#include "ThreadPoolExecutor.hpp"

constexpr auto softening_factor = 0.5;   // Adjustment to Power softening.
constexpr auto tolerance_factor = 0.25;  // Ratio of tolerance to softening.

template <ODEState OutputStateType>
struct MostRecentOutput {
  using value_type = typename ode_state_traits<OutputStateType>::value_type;
  value_type time{};
  OutputStateType state{};

  template <ODEState StateType>
  auto save_state(value_type t, StateType x) {
    time = t;
    x.copy_to_span(span(state));
  }
};

template <int NThreads>
struct ThreadPool : public ThreadPoolExecutor {
  ThreadPool() : ThreadPoolExecutor(NThreads) {}
};

template <std::size_t N, typename Integrator, typename ParallelExecutor,
          template <typename, std::size_t> typename StateAllocator,
          std::floating_point ValueType>
static void BM_nbody_rk_simulator(benchmark::State& state) {
  auto scenario = SpinningParticlesInBox<N, StateAllocator, ValueType>{
      softening_factor, tolerance_factor};
  static constexpr auto n_var = scenario.n_var;

  auto exe = ParallelExecutor{};
  auto output = MostRecentOutput<HeapState<ValueType, n_var>>{};

  auto t0 = 0.0;
  auto tf = scenario.tf;

  for (auto _ : state) {
    Integrator::integrate(scenario.initial_state, t0, tf,
                          scenario.tolerance_array, scenario.tolerance_array,
                          NBodyODE<ValueType, n_var>{scenario.softening},
                          output, exe);
    benchmark::DoNotOptimize(output.time);
    benchmark::DoNotOptimize(output.state);
  }

  state.SetItemsProcessed(state.iterations() * N);
}

#define BENCHMARK_SET(N)                                                     \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTHE21>,   \
                     SingleThreadedExecutor, HeapState, double)              \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF45>,  \
                     SingleThreadedExecutor, HeapState, double)              \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDOPRI5>, \
                     SingleThreadedExecutor, HeapState, double)              \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDVERK>,  \
                     SingleThreadedExecutor, HeapState, double)              \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF78>,  \
                     SingleThreadedExecutor, HeapState, double)              \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTHE21>,   \
                     ThreadPool<4>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF45>,  \
                     ThreadPool<4>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDOPRI5>, \
                     ThreadPool<4>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDVERK>,  \
                     ThreadPool<4>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF78>,  \
                     ThreadPool<4>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTHE21>,   \
                     ThreadPool<8>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF45>,  \
                     ThreadPool<8>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDOPRI5>, \
                     ThreadPool<8>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDVERK>,  \
                     ThreadPool<8>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF78>,  \
                     ThreadPool<8>, HeapState, double)                       \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTHE21>,   \
                     ThreadPool<12>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF45>,  \
                     ThreadPool<12>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDOPRI5>, \
                     ThreadPool<12>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDVERK>,  \
                     ThreadPool<12>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF78>,  \
                     ThreadPool<12>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTHE21>,   \
                     ThreadPool<16>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF45>,  \
                     ThreadPool<16>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDOPRI5>, \
                     ThreadPool<16>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDVERK>,  \
                     ThreadPool<16>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF78>,  \
                     ThreadPool<16>, HeapState, double)                      \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTHE21>,   \
                     CudaExecutor, CudaState, double)                        \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF45>,  \
                     CudaExecutor, CudaState, double)                        \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDOPRI5>, \
                     CudaExecutor, CudaState, double)                        \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTDVERK>,  \
                     CudaExecutor, CudaState, double)                        \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();                                                       \
  BENCHMARK_TEMPLATE(BM_nbody_rk_simulator, N, RKEmbeddedParallel<BTRKF78>,  \
                     CudaExecutor, CudaState, double)                        \
      ->MeasureProcessCPUTime()                                              \
      ->UseRealTime();

BENCHMARK_SET(64)
BENCHMARK_SET(256)
BENCHMARK_SET(1024)
BENCHMARK_SET(4096)
BENCHMARK_SET(16384)
BENCHMARK_SET(65536)
