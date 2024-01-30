#include <ranges>
#include <span>

template <typename StateElement, typename StepType, typename DerivFunc>
auto euler_step(std::span<StateElement> state, StepType dt, DerivFunc f)
{
    auto const& derivative = f();
    for (auto&& [x, dx_dt] : std::views::zip(state, derivative))
    {
        x += dx_dt * dt;
    }
}