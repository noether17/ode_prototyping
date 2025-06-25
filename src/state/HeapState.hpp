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
  static constexpr auto size() { return N; }

  HeapState() : state_array_{std::make_unique<T[]>(N)} {}
  explicit HeapState(std::span<T const, N> state_span)
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

  friend auto span(HeapState& state) {
    return std::span<T, N>{state.data(), state.size()};
  }
  friend auto span(HeapState const& state) {
    return std::span<T const, N>{state.data(), state.size()};
  }

  auto& operator[](std::size_t i) { return state_array_[i]; }
  auto const& operator[](std::size_t i) const { return state_array_[i]; }

 private:
  std::unique_ptr<T[]> state_array_{};
};

template <typename R>
HeapState(R&&) -> HeapState<typename std::remove_reference_t<R>::value_type,
                            std::tuple_size_v<std::remove_reference_t<R>>>;
