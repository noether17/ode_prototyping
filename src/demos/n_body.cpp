#include <algorithm>
#include <array>
#include <fstream>

#include "BTRKF78.hpp"
#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "RKEmbeddedParallel.hpp"
#include "RawOutput.hpp"
#include "ThreadPoolExecutor.hpp"

int main() {
  // auto x0_data =
  //     std::array{1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5,
  //     0.0};
  // auto x0_data =
  //    std::array{0.9700436,   -0.24308753, 0.0, -0.9700436,  0.24308753,  0.0,
  //               0.0,         0.0,         0.0, 0.466203685, 0.43236573,  0.0,
  //               0.466203685, 0.43236573,  0.0, -0.93240737, -0.86473146,
  //               0.0};
  // auto x0_data = std::array{1.0, 3.0, 0.0, -2.0, -1.0, 0.0, 1.0, -1.0, 0.0,
  //                          0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0,  0.0};
  auto x0_data =
      std::array{1.657666,  0.0,       0.0, 0.439775,  -0.169717, 0.0,
                 -1.268608, -0.267651, 0.0, -1.268608, 0.267651,  0.0,
                 0.439775,  0.169717,  0.0, 0.0,       -0.593786, 0.0,
                 1.822785,  0.128248,  0.0, 1.271564,  0.168645,  0.0,
                 -1.271564, 0.168645,  0.0, -1.822785, 0.128248,  0.0};
  auto constexpr n_var = x0_data.size();

  auto thread_pool = ThreadPoolExecutor(8);
  auto integrator =
      RKEmbeddedParallel<HeapState, double, n_var, BTRKF78, NBodyODE<n_var>,
                         RawOutput<HeapState<double, n_var>>,
                         ThreadPoolExecutor>{};
  auto output = RawOutput<HeapState<double, n_var>>{};

  auto x0 = HeapState<double, n_var>{x0_data};
  auto t0 = 0.0;
  auto tf = 6.3;
  auto tol = HeapState<double, n_var>{};
  std::fill(tol.data(), tol.data() + n_var, 1.0e-10);

  integrator.integrate(x0, t0, tf, tol, tol, NBodyODE<n_var>{}, output,
                       thread_pool);

  auto output_file = std::ofstream{"RKF78_threadpool_n_body_output.txt"};
  for (std::size_t i = 0; i < output.times.size(); ++i) {
    output_file << output.times[i];
    for (std::size_t j = 0; j < n_var; ++j) {
      output_file << ',' << output.states[i][j];
    }
    output_file << '\n';
  }

  return 0;
}
