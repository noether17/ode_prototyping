#include <array>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "BTRKF78.hpp"
#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "ParticlesInBox.hpp"
#include "RKEmbeddedParallel.hpp"
#include "RawOutput.hpp"
#include "SpinningParticlesInBox.hpp"
#include "ThreadPoolExecutor.hpp"

template <typename IntegrationMethod, typename ParallelizationMethod>
auto generate_filename(auto const& scenario) {
  // scenario name and number of particles
  auto filename = scenario.name + '_' + std::to_string(scenario.n_particles);

  // integration method
  if constexpr (std::is_same_v<IntegrationMethod, BTRKF78>) {
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

void output_to_file(std::string const& filename, auto const& output,
                    double softening) {
  auto const n_times = static_cast<std::size_t>(output.times.size());
  auto const n_var = static_cast<std::size_t>(output.states[0].size());

  auto output_file = std::ofstream{filename, std::ios::out | std::ios::binary};
  output_file.write(reinterpret_cast<char const*>(&n_times), sizeof(n_times));
  output_file.write(reinterpret_cast<char const*>(&n_var), sizeof(n_var));
  output_file.write(reinterpret_cast<char const*>(&softening),
                    sizeof(softening));
  for (std::size_t i = 0; i < n_times; ++i) {
    output_file.write(reinterpret_cast<char const*>(&output.times[i]),
                      sizeof(output.times[i]));
    for (std::size_t j = 0; j < n_var; ++j) {
      output_file.write(reinterpret_cast<char const*>(&output.states[i][j]),
                        sizeof(output.states[i][j]));
    }
  }
}

template <int N>
void run_threadpool_scenario(double softening_divisor, double tolerance) {
  constexpr auto n_repetitions = 3;
  for (auto i = 0; i < n_repetitions; ++i) {
    auto scenario = SpinningParticlesInBox<N, HeapState, double>{
        softening_divisor, tolerance};
    constexpr auto n_var = scenario.n_var;

    auto tp_exe = ThreadPoolExecutor{12};
    auto integrator = RKEmbeddedParallel<
        HeapState, double, n_var, BTRKF78, NBodyODE<double, n_var>,
        RawOutput<HeapState<double, n_var>>, ThreadPoolExecutor>{};
    auto output = RawOutput<HeapState<double, n_var>>{};

    auto t0 = 0.0;
    auto tf = scenario.tf;

    integrator.integrate(scenario.initial_state, t0, tf,
                         scenario.tolerance_array, scenario.tolerance_array,
                         NBodyODE<double, n_var>{scenario.softening}, output,
                         tp_exe);

    auto filename = generate_filename<BTRKF78, ThreadPoolExecutor>(scenario);
    output_to_file(filename, output, scenario.softening);

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(
        1s);  // crude solution to make sure filenames are distinct.
  }
}

template <int N>
void run_cuda_scenario(double softening_divisor, double tolerance) {
  constexpr auto n_repetitions = 3;
  for (auto i = 0; i < n_repetitions; ++i) {
    auto scenario = SpinningParticlesInBox<N, CudaState, double>{
        softening_divisor, tolerance};
    constexpr auto n_var = scenario.n_var;

    auto cuda_exe = CudaExecutor{};
    auto integrator =
        RKEmbeddedParallel<CudaState, double, n_var, BTRKF78,
                           NBodyODE<double, n_var>,
                           RawOutput<HeapState<double, n_var>>, CudaExecutor>{};
    auto output = RawOutput<HeapState<double, n_var>>{};

    auto t0 = 0.0;
    auto tf = scenario.tf;

    integrator.integrate(scenario.initial_state, t0, tf,
                         scenario.tolerance_array, scenario.tolerance_array,
                         NBodyODE<double, n_var>{scenario.softening}, output,
                         cuda_exe);

    auto filename = generate_filename<BTRKF78, CudaExecutor>(scenario);
    output_to_file(filename, output, scenario.softening);

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(
        1s);  // crude solution to make sure filenames are distinct.
  }
}

int main() {
  for (auto softening_divisor : {1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0}) {
    std::cout << "Starting with softening divisor " << softening_divisor
              << '\n';
    for (auto tolerance_value : {1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6,
                                 1.0e-6, 3.0e-7, 1.0e-7}) {
      std::cout << "  Starting with tolerance value " << tolerance_value
                << '\n';
      run_threadpool_scenario<64>(softening_divisor, tolerance_value);
    }
  }

  return 0;
}
