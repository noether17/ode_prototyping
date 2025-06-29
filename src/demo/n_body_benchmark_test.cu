#include <array>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <type_traits>

#include "BTRKF78.hpp"
#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "HeapState.hpp"
#include "NBodyODE.hpp"
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
void run_threadpool_scenario(double softening_divisor,
                             double tolerance_factor) {
  constexpr auto n_repetitions = 3;
  for (auto i = 0; i < n_repetitions; ++i) {
    auto scenario = SpinningParticlesInBox<N, HeapState, double>{
        softening_divisor, tolerance_factor};
    constexpr auto n_var = scenario.n_var;

    auto tp_exe = ThreadPoolExecutor{12};
    auto integrator = RKEmbeddedParallel<BTRKF78, NBodyODE<double, n_var>>{};
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
void run_cuda_scenario(double softening_divisor, double tolerance_factor) {
  constexpr auto n_repetitions = 1;
  for (auto i = 0; i < n_repetitions; ++i) {
    auto scenario = SpinningParticlesInBox<N, CudaState, double>{
        softening_divisor, tolerance_factor};
    constexpr auto n_var = scenario.n_var;

    auto cuda_exe = CudaExecutor{};
    auto integrator = RKEmbeddedParallel<BTRKF78, NBodyODE<double, n_var>>{};
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

template <int N>
void do_multiple_scenario_run() {
  std::cout << "Starting scenarios with N = " << N << '\n';
  auto start = std::chrono::high_resolution_clock::now();
  for (auto softening_divisor : {0.5, 1.0, 2.0, 4.0, 8.0, 16.0}) {
    std::cout << "  Starting scenarios with softening divisor "
              << softening_divisor << '\n';
    for (auto tolerance_factor : {2.0, 1.0, 0.5, 0.25, 0.125, 0.0625}) {
      std::cout << "    Starting scenario with tolerance factor "
                << tolerance_factor << '\n';
      run_cuda_scenario<N>(softening_divisor, tolerance_factor);
    }
  }
  auto duration = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now() - start);
  std::cout << "Completed scenarios with N = " << N
            << ". Scenarios completed in " << duration.count() << "s\n";
}

int main() {
  do_multiple_scenario_run<64>();
  do_multiple_scenario_run<256>();
  do_multiple_scenario_run<1024>();
  do_multiple_scenario_run<4096>();
  do_multiple_scenario_run<16384>();
  do_multiple_scenario_run<65536>();
}
