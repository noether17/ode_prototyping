#include <concepts>
#include <ranges>

template <std::ranges::random_access_range StateType, std::floating_point StepType, typename DerivFunc>
auto euler_step(StateType& state, StepType dt, DerivFunc f)
{
    auto const& derivative = f();
    for (auto&& [x, dx_dt] : std::views::zip(state, derivative))
    {
        x += dx_dt * dt;
    }
}