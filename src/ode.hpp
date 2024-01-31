#include <concepts>
#include <ranges>

namespace rng = std::ranges;

template <typename Callable, typename StateType>
concept DerivativeFunc = std::invocable<Callable> &&
    rng::random_access_range<std::invoke_result_t<Callable>> &&
    rng::random_access_range<StateType> &&
    std::same_as<std::ranges::range_value_t<std::invoke_result_t<Callable>>,
        std::ranges::range_value_t<StateType>>;

auto euler_step(rng::random_access_range auto& state, auto f, std::floating_point auto dt)
requires DerivativeFunc<decltype(f), decltype(state)>
{
    auto const& derivative = f();
    for (auto&& [x, dx_dt] : std::views::zip(state, derivative))
    {
        x += dx_dt * dt;
    }
}