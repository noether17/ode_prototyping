#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

template <typename T>
concept ODEState = requires {
  typename T::value_type;
} && std::floating_point<typename T::value_type> && requires(T t) {
  { t.size() } -> std::same_as<std::size_t>;
  { t.data() } -> std::same_as<typename T::value_type*>;
  { std::as_const(t).data() } -> std::same_as<typename T::value_type const*>;
} && (requires { std::integral_constant<std::size_t, T{}.size()>{}; } ||
      requires { std::integral_constant<std::size_t, T::size()>{}; });

template <typename T>
struct state_traits;

template <template <typename, std::size_t> typename StateType, typename T,
          std::size_t N>
  requires ODEState<StateType<T, N>>
struct state_traits<StateType<T, N>> {
  static constexpr auto size = N;
  template <std::size_t M>
  using resized_state_type = StateType<T, M>;
};
