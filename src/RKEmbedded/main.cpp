#include <array>
#include <chrono>
#include <iostream>

#include "AllocatedState.hpp"
#include "BTDOPRI5.hpp"
#include "BTDVERK.hpp"
#include "BTRKF45.hpp"
#include "BTRKF78.hpp"
#include "RKEmbedded.hpp"
#include "RawOutput.hpp"

auto ode(AllocatedState<2> const& x, AllocatedState<2>& dxdt) {
  dxdt[0] = 1.0;
  dxdt[1] = -x[1] * x[1] * x[1] + std::sin(x[0]);
}

using OutputType = RawOutput<AllocatedState<2>>;

template <typename ButcherTableau>
using SingleThreadedIntegrator =
    RKEmbeddedSingleThreaded<AllocatedState<2>, ButcherTableau, decltype(ode),
                             OutputType>;

int main() {
  auto const x0 = AllocatedState<2>(std::array{0.0, 0.0});
  auto constexpr tol = 1.0e-10;
  auto const atol = AllocatedState<2>(std::array{tol, tol});
  auto const rtol = AllocatedState<2>(std::array{tol, tol});

  auto integrator_RKF45 = SingleThreadedIntegrator<BTRKF45>{};
  auto output = OutputType{};
  auto start = std::chrono::high_resolution_clock::now();
  integrator_RKF45.integrate(x0, 0.0, 10.0, atol, rtol, ode, output);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "RKF45 results -- ";
  std::cout << "execution time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us; ";
  std::cout << "steps: " << output.times.size() << "; ";
  std::cout << "finish: (" << output.times.back() << ", "
            << output.states.back()[1] << ")\n";
  // for (auto const& [t, x] :
  //      vws::zip(integrator_RKF45.times, integrator_RKF45.states)) {
  //   std::cout << '(' << t << ", " << x[1] << ')';
  // }
  // std::cout << '\n';

  auto integrator_DOPRI5 = SingleThreadedIntegrator<BTDOPRI5>{};
  output = OutputType{};
  start = std::chrono::high_resolution_clock::now();
  integrator_DOPRI5.integrate(x0, 0.0, 10.0, atol, rtol, ode, output);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "DOPRI5 results -- ";
  std::cout << "execution time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us; ";
  std::cout << "steps: " << output.times.size() << "; ";
  std::cout << "finish: (" << output.times.back() << ", "
            << output.states.back()[1] << ")\n";
  // for (auto const& [t, x] :
  //      vws::zip(integrator_DOPRI5.times, integrator_DOPRI5.states)) {
  //   std::cout << '(' << t << ", " << x[1] << ')';
  // }
  // std::cout << '\n';

  auto integrator_DVERK = SingleThreadedIntegrator<BTDVERK>{};
  output = OutputType{};
  start = std::chrono::high_resolution_clock::now();
  integrator_DVERK.integrate(x0, 0.0, 10.0, atol, rtol, ode, output);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "DVERK results -- ";
  std::cout << "execution time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us; ";
  std::cout << "steps: " << output.times.size() << "; ";
  std::cout << "finish: (" << output.times.back() << ", "
            << output.states.back()[1] << ")\n";
  // for (auto const& [t, x] :
  //      vws::zip(integrator_DVERK.times, integrator_DVERK.states)) {
  //   std::cout << '(' << t << ", " << x[1] << ')';
  // }
  // std::cout << '\n';

  auto integrator_RKF78 = SingleThreadedIntegrator<BTRKF78>{};
  output = OutputType{};
  start = std::chrono::high_resolution_clock::now();
  integrator_RKF78.integrate(x0, 0.0, 10.0, atol, rtol, ode, output);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "RKF78 results -- ";
  std::cout << "execution time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us; ";
  std::cout << "steps: " << output.times.size() << "; ";
  std::cout << "finish: (" << output.times.back() << ", "
            << output.states.back()[1] << ")\n";
  // for (auto const& [t, x] :
  //      vws::zip(integrator_RKF78.times, integrator_RKF78.states)) {
  //   std::cout << '(' << t << ", " << x[1] << ')';
  // }
  // std::cout << '\n';
}
