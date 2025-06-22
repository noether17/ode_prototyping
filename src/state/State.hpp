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
      requires { std::integral_constant<std::size_t, T::size()>{}; }) &&
requires {
  typename T::span_type;
  typename T::const_span_type;
} && requires(T t) {
  { span(t) } -> std::same_as<typename T::span_type>;
  { span(std::as_const(t)) } -> std::same_as<typename T::const_span_type>;
};

template <typename T>
struct state_traits;

template <template <typename, std::size_t> typename StateManager, typename T,
          std::size_t N>
  requires ODEState<StateManager<T, N>>
struct state_traits<StateManager<T, N>> {
  using value_type = T;
  static constexpr auto size = N;
};

template <typename T, std::size_t M>
struct resized_ode_state;

template <template <typename, std::size_t> typename StateManager, typename T,
          std::size_t N, std::size_t M>
  requires ODEState<StateManager<T, N>>
struct resized_ode_state<StateManager<T, N>, M> {
  using type = StateManager<T, M>;
};

template <ODEState StateType, std::size_t M>
using ResizedODEState = resized_ode_state<StateType, M>::type;
