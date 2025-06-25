#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

template <typename T>
class ode_state_traits;

template <template <typename, std::size_t> typename StateManager, typename T,
          std::size_t N>
class ode_state_traits<StateManager<T, N>> {
  template <typename U>
  static auto deduce_span(U&& u) -> decltype(span(std::forward<U>(u)));

 public:
  using type = StateManager<T, N>;
  using value_type = T;
  static constexpr auto size = N;

  using span_type = decltype(deduce_span(std::declval<type&>()));
  using const_span_type = decltype(deduce_span(std::declval<const type&>()));
};

template <typename T>
concept ODEState = requires {
  typename ode_state_traits<T>;
} && std::floating_point<typename ode_state_traits<T>::value_type> &&
requires(T t) {
  { t.size() } -> std::same_as<std::size_t>;
  { t.data() } -> std::same_as<typename ode_state_traits<T>::value_type*>;
  { std::as_const(t).data() } ->
    std::same_as<typename ode_state_traits<T>::value_type const*>;
} && (requires { std::integral_constant<std::size_t, T{}.size()>{}; } ||
      requires { std::integral_constant<std::size_t, T::size()>{}; }) &&
requires {
  typename ode_state_traits<T>::span_type;
  typename ode_state_traits<T>::const_span_type;
} && requires(T t) {
  { span(t) } -> std::same_as<typename ode_state_traits<T>::span_type>;
  { span(std::as_const(t)) } ->
    std::same_as<typename ode_state_traits<T>::const_span_type>;
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
