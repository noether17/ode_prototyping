#include <concepts>
#include <ranges>

template <typename Callable, typename StateType>
concept DerivativeFunc = std::invocable<Callable> &&
    std::ranges::random_access_range<std::invoke_result_t<Callable>> &&
    std::ranges::random_access_range<StateType> &&
    std::convertible_to<std::invoke_result_t<Callable>, StateType>;

template <std::ranges::random_access_range StateType, std::floating_point StepType, DerivativeFunc<StateType> F>
auto euler_step(StateType& state, StepType dt, F f)
{
    auto const& derivative = f();
    for (auto&& [x, dx_dt] : std::views::zip(state, derivative))
    {
        x += dx_dt * dt;
    }
}