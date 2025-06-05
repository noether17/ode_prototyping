#pragma once

#include <memory>
#include <span>
#include <utility>

template <template <typename, std::size_t> typename ContainerType, typename T,
          std::size_t N>
class HeapState {
 public:
  template <typename ValueType, std::size_t Size>
  using container_type = ContainerType<ValueType, Size>;
  using state_type = container_type<T, N>;
  using value_type = T;

  HeapState() : state_{std::make_unique<state_type>()} {}
  explicit HeapState(state_type const& state)
      : state_{std::make_unique<state_type>(state)} {}
  explicit HeapState(std::span<value_type const, N> state)
      : state_{std::make_unique<state_type>()} {
    std::copy(state.begin(), state.end(), state_->begin());
  }
  HeapState(HeapState const& state)
      : state_{std::make_unique<state_type>(*state.state_)} {}
  HeapState(HeapState&& state) = default;
  auto& operator=(HeapState const& state) {
    if (this != &state) {
      state_ = std::make_unique<state_type>(*state.state_);
    }
    return *this;
  }
  auto& operator=(HeapState&& state) {
    state_ = std::move(state.state_);
    return *this;
  }
  ~HeapState() = default;

  auto& operator[](int i) { return (*state_)[i]; }
  auto const& operator[](int i) const { return (*state_)[i]; }

  auto* data() { return state_->data(); }
  auto const* data() const { return state_->data(); }

  operator std::span<value_type, N>() { return *state_; }
  operator std::span<value_type const, N>() const { return *state_; }

  static constexpr auto size() { return N; }

 private:
  std::unique_ptr<state_type> state_{};
};
