#pragma once

#include <algorithm>
#include <concepts>
#include <memory>
#include <span>
#include <tuple>
#include <type_traits>

template <std::floating_point T, std::size_t N>
class HeapState {
 public:
  using value_type = T;
  using span_type = std::span<T, N>;
  using const_span_type = std::span<T const, N>;
  static constexpr auto size() { return N; }

  HeapState() : state_array_{std::make_unique<T[]>(N)} {}
  explicit HeapState(const_span_type state_span)
      : state_array_{std::make_unique_for_overwrite<T[]>(N)} {
    std::ranges::copy(state_span, state_array_.get());
  }
  HeapState(HeapState const& other)
      : state_array_{std::make_unique_for_overwrite<T[]>(N)} {
    std::copy(other.state_array_.get(), other.state_array_.get() + N,
              state_array_.get());
  }
  HeapState(HeapState&& other) noexcept
      : state_array_{std::move(other.state_array_)} {
    // Need to preserve class invariant of representing fixed-size array, but no
    // guarantee of default initialization.
    other.state_array_ = std::make_unique_for_overwrite<T[]>(N);
  }
  HeapState& operator=(HeapState const& other) {
    if (this != &other) {
      std::copy(other.state_array_.get(), other.state_array_.get() + N,
                state_array_.get());
    }
    return *this;
  }
  HeapState& operator=(HeapState&& other) noexcept {
    std::swap(state_array_, other.state_array_);
    return *this;
  }
  ~HeapState() = default;

  auto* data() { return state_array_.get(); }
  auto const* data() const { return state_array_.get(); }

  operator span_type() { return span_type{data(), size()}; }
  operator const_span_type() const { return const_span_type{data(), size()}; }

  friend auto span(HeapState& state) { return span_type{state}; }
  friend auto span(HeapState const& state) { return const_span_type{state}; }

  auto& operator[](std::size_t i) { return state_array_[i]; }
  auto const& operator[](std::size_t i) const { return state_array_[i]; }

 private:
  std::unique_ptr<T[]> state_array_{};
};

template <typename R>
HeapState(R&&) -> HeapState<typename std::remove_reference_t<R>::value_type,
                            std::tuple_size_v<std::remove_reference_t<R>>>;
