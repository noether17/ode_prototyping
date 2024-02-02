#include <concepts>
#include <ranges>

namespace rng = std::ranges;

//template <typename Callable, typename StateType>
//concept DerivativeFunc = std::invocable<Callable> &&
//    rng::random_access_range<std::invoke_result_t<Callable>> &&
//    rng::random_access_range<StateType> &&
//    std::same_as<std::ranges::range_value_t<std::invoke_result_t<Callable>>,
//        std::ranges::range_value_t<StateType>>;

template <std::size_t... I>
auto euler_step_impl(std::floating_point auto dt, auto&& state_tuple, std::index_sequence<I...>)
{
    auto increment = [dt](auto&& state, auto&& derivative)
    {
        for (auto&& [x, dx_dt] : std::views::zip(state, derivative))
        {
            x += dx_dt*dt;
        }
    };
    (..., increment(std::get<I>(state_tuple), std::get<I + 1>(state_tuple)));
}

auto euler_step(std::floating_point auto dt, rng::random_access_range auto& state,
    rng::random_access_range auto&&... derivatives)
{
    euler_step_impl(dt, std::forward_as_tuple(state, derivatives...),
        std::make_index_sequence<sizeof...(derivatives)>{});
}