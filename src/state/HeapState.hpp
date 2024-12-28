#pragma once

#include <array>
#include <memory>
#include <span>
#include <utility>

template <typename ValueType, int N>
class HeapState {
 public:
  using StateType = std::array<ValueType, N>;
  using value_type = ValueType;
  HeapState() : state_{std::make_unique<StateType>()} {}
  explicit HeapState(std::span<ValueType const, N> state)
      : state_{std::make_unique<StateType>()} {
    std::copy(state.begin(), state.end(), state_->begin());
  }
  HeapState(HeapState const& state)
      : state_{std::make_unique<StateType>(*state.state_)} {}
  HeapState(HeapState&& state) = default;
  auto& operator=(HeapState const& state) {
    if (this != &state) {
      state_ = std::make_unique<StateType>(*state.state_);
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

  operator std::span<ValueType, N>() { return *state_; }
  operator std::span<ValueType const, N>() const { return *state_; }

  auto static constexpr size() { return N; }

 private:
  std::unique_ptr<StateType> state_{};
};
