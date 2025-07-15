#include <array>
#include <charconv>
#include <chrono>
#include <iomanip>
#include <string>
#include <thread>
#include <type_traits>

#include "BTDOPRI5.hpp"
#include "BTDVERK.hpp"
#include "BTHE21.hpp"
#include "BTRKF45.hpp"
#include "BTRKF78.hpp"
#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "RKEmbeddedParallel.hpp"
#include "RawOutputWithLog.hpp"
#include "SpinningParticlesInBox.hpp"
#include "ThreadPoolExecutor.hpp"
#include "nbody_io.hpp"

template <typename Integrator>
struct butcher_tableau;

template <typename ButcherTableau>
struct butcher_tableau<RKEmbeddedParallel<ButcherTableau>> {
  using type = ButcherTableau;
};

template <typename Integrator, typename ParallelizationMethod>
auto generate_filename(auto const& scenario) {
  // scenario name and number of particles
  auto filename = scenario.name + '_' + std::to_string(scenario.n_particles);

  // integration method
  using BT = typename butcher_tableau<Integrator>::type;
  if constexpr (std::is_same_v<BT, BTHE21>) {
    filename += "_HE21";
  } else if constexpr (std::is_same_v<BT, BTRKF45>) {
    filename += "_RKF45";
  } else if constexpr (std::is_same_v<BT, BTDOPRI5>) {
    filename += "_DOPRI5";
  } else if constexpr (std::is_same_v<BT, BTDVERK>) {
    filename += "_DVERK";
  } else if constexpr (std::is_same_v<BT, BTRKF78>) {
    filename += "_RKF78";
  } else {
    throw "Unrecognized integration method!\n";
  }

  // parallelization method
  if constexpr (std::is_same_v<ParallelizationMethod, CudaExecutor>) {
    filename += "_CUDA";
  } else if constexpr (std::is_same_v<ParallelizationMethod,
                                      ThreadPoolExecutor>) {
    filename += "_ThreadPool";
  } else {
    throw "Unrecognized parallelization method!\n";
  }

  // current time
  auto now = std::chrono::system_clock::now();
  auto tt = std::chrono::system_clock::to_time_t(now);
  auto tss = std::ostringstream{};
  tss << std::put_time(std::localtime(&tt), "_%Y%m%d_%H%M%S");
  filename += tss.str();

  // softening and tolerance
  constexpr auto buffer_size = 8;
  filename += "_sof_";
  auto sof_str = std::array<char, buffer_size>{'\0'};
  std::to_chars(sof_str.begin(), sof_str.end(), scenario.softening,
                std::chars_format::scientific, 1);
  filename += sof_str.data();
  filename += "_tol_";
  auto tol_str = std::array<char, buffer_size>{'\0'};
  std::to_chars(tol_str.begin(), tol_str.end(), scenario.tolerance_value,
                std::chars_format::scientific, 1);
  filename += tol_str.data();

  // extension
  filename += ".bin";

  return filename;
}

template <int N>
void run_threadpool_scenario(double softening_factor, double tolerance_factor) {
  constexpr auto n_repetitions = 3;
  for (auto i = 0; i < n_repetitions; ++i) {
    auto scenario = SpinningParticlesInBox<N, HeapState, double>{
        softening_factor, tolerance_factor};
    constexpr auto n_var = scenario.n_var;

    auto tp_exe = ThreadPoolExecutor{12};
    auto output = RawOutputWithLog<HeapState<double, n_var>>{};

    auto t0 = 0.0;
    auto tf = scenario.tf;

    RKEmbeddedParallel<BTRKF78>::integrate(
        scenario.initial_state, t0, tf, scenario.tolerance_array,
        scenario.tolerance_array, NBodyODE<double, n_var>{scenario.softening},
        output, tp_exe);

    auto filename = generate_filename<BTRKF78, ThreadPoolExecutor>(scenario);
    output_to_file(filename, output, scenario.softening);

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(
        1s);  // crude solution to make sure filenames are distinct.
  }
}

template <int N, typename Integrator>
void run_cuda_scenario(double softening_factor, double tolerance_factor) {
  constexpr auto n_repetitions = 1;
  for (auto i = 0; i < n_repetitions; ++i) {
    auto scenario = SpinningParticlesInBox<N, CudaState, double>{
        softening_factor, tolerance_factor};
    constexpr auto n_var = scenario.n_var;

    auto cuda_exe = CudaExecutor{};
    auto output = RawOutputWithLog<HeapState<double, n_var>>{};

    auto t0 = 0.0;
    auto tf = scenario.tf;

    Integrator::integrate(scenario.initial_state, t0, tf,
                          scenario.tolerance_array, scenario.tolerance_array,
                          NBodyODE<double, n_var>{scenario.softening}, output,
                          cuda_exe);

    auto filename = generate_filename<Integrator, CudaExecutor>(scenario);
    output_to_file(filename, output, scenario.softening);

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(
        1s);  // crude solution to make sure filenames are distinct.
  }
}

template <int N, typename Integrator>
void do_multiple_scenario_run() {
  using BT = typename butcher_tableau<Integrator>::type;
  std::cout << "Starting scenarios using " << typeid(BT).name()
            << " with N = " << N << '\n';
  auto start = std::chrono::high_resolution_clock::now();
  for (auto softening_factor : {0.5}) {
    std::cout << "  Starting scenarios with softening factor "
              << softening_factor << '\n';
    for (auto tolerance_factor : {0.5, 0.25, 0.125, 0.0625}) {
      std::cout << "    Starting scenario with tolerance factor "
                << tolerance_factor << '\n';
      run_cuda_scenario<N, Integrator>(softening_factor, tolerance_factor);
    }
  }
  auto duration = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start);
  std::cout << "Completed scenarios with N = " << N
            << ". Scenarios completed in " << duration.count() << "s\n";
}

#define DO_MULTIPLE_SCENARIO_ALL_INTEGRATOR_RUN(N)             \
  do_multiple_scenario_run<N, RKEmbeddedParallel<BTHE21>>();   \
  do_multiple_scenario_run<N, RKEmbeddedParallel<BTRKF45>>();  \
  do_multiple_scenario_run<N, RKEmbeddedParallel<BTDOPRI5>>(); \
  do_multiple_scenario_run<N, RKEmbeddedParallel<BTDVERK>>();  \
  do_multiple_scenario_run<N, RKEmbeddedParallel<BTRKF78>>();

int main() {
  DO_MULTIPLE_SCENARIO_ALL_INTEGRATOR_RUN(256);
  DO_MULTIPLE_SCENARIO_ALL_INTEGRATOR_RUN(1024);
  DO_MULTIPLE_SCENARIO_ALL_INTEGRATOR_RUN(4096);
  DO_MULTIPLE_SCENARIO_ALL_INTEGRATOR_RUN(16384);
  DO_MULTIPLE_SCENARIO_ALL_INTEGRATOR_RUN(65536);
  DO_MULTIPLE_SCENARIO_ALL_INTEGRATOR_RUN(262144);
}
