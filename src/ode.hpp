#include <concepts>
#include <ranges>

namespace rng = std::ranges;
namespace vws = std::views;

template <std::size_t... I>
auto euler_step_impl(std::floating_point auto dt, auto&& state_tuple,
                     std::index_sequence<I...>) {
  auto increment = [dt](auto&& state, auto&& derivative) {
    for (auto&& [x, dx_dt] : vws::zip(state, derivative)) {
      x += dx_dt * dt;
    }
  };
  (..., increment(std::get<I>(state_tuple), std::get<I + 1>(state_tuple)));
}

auto euler_step(std::floating_point auto dt,
                rng::random_access_range auto& state,
                rng::random_access_range auto&&... derivatives) {
  euler_step_impl(dt, std::forward_as_tuple(state, derivatives...),
                  std::make_index_sequence<sizeof...(derivatives)>{});
}

auto integrate_euler_fixed(std::floating_point auto ti,
                           std::floating_point auto tf,
                           std::floating_point auto dt, auto&& f,
                           rng::random_access_range auto&&... states) {
  for (auto t = ti; t < tf; t += dt) {
    euler_step(dt, states..., f());
  }
}