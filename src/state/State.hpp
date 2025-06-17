#pragma once

#include <concepts>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

template <typename T>
concept ODEState = requires {
  typename T::value_type;
} && std::floating_point<typename T::value_type> && requires(T t) {
  { t.size() } -> std::same_as<std::size_t>;
  std::integral_constant<std::size_t, T{}.size()>{};
  { t.data() } -> std::same_as<typename T::value_type*>;
  { std::as_const(t).data() } -> std::same_as<typename T::value_type const*>;
};

template <typename T>
concept StateArray = requires {
  typename T::value_type;
} && std::floating_point<typename T::value_type> && requires(T t) {
  { t.size() } -> std::same_as<std::size_t>;
} && std::convertible_to<T, std::span<typename T::value_type, T{}.size()>>;

template <template <std::ranges::contiguous_range> typename StateManager,
          template <std::floating_point, std::size_t> typename Container,
          std::floating_point ValueType, std::size_t Size>
struct State : public StateManager<Container<ValueType, Size>> {
  static constexpr auto size() { return Size; }

  template <typename T>
  using manager = StateManager<T>;
  template <typename T, std::size_t N>
  using container = Container<T, N>;
  using state_type = container<ValueType, Size>;
  using value_type = ValueType;
  using base = manager<state_type>;
  using base::base;

  using base::operator std::span<value_type, Size>;
  using base::operator std::span<value_type const, Size>;

  State(state_type const& state) : base{state} {}

  auto* data() { return std::span{*this}.data(); }
  auto const* data() const { return std::span{*this}.data(); }
};
