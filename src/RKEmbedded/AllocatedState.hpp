#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <span>

#include "ODEState.hpp"

template <int N>
class AllocatedState : public ODEState<AllocatedState<N>> {
 public:
  using value_type = double;
  AllocatedState() : state_{std::make_unique<std::array<double, N>>()} {}
  AllocatedState(std::span<double const, N> state)
      : state_{std::make_unique<std::array<double, N>>()} {
    std::copy(state.begin(), state.end(), state_->begin());
  }
  AllocatedState(AllocatedState const& v)
      : state_{std::make_unique<std::array<double, N>>(*v.state_)} {}
  AllocatedState(AllocatedState&& v) = default;
  auto& operator=(AllocatedState const& v) {
    if (this != &v) {
      state_ = std::make_unique<std::array<double, N>>(*v.state_);
    }
    return *this;
  }
  auto& operator=(AllocatedState&& v) {
    state_ = std::move(v.state_);
    return *this;
  }
  ~AllocatedState() = default;

  auto& operator[](int i) { return (*state_)[i]; }
  auto const& operator[](int i) const { return (*state_)[i]; }

  auto* data() { return state_->data(); }
  auto const* data() const { return state_->data(); }

  auto static constexpr size() { return N; }

  template <typename UnaryOp>
  friend void elementwise_unary_op(AllocatedState& v, UnaryOp unary_op) {
    for (auto i = 0; i < N; ++i) {
      unary_op(v[i]);
    }
  }

  template <typename UnaryOp>
  friend void elementwise_unary_op(AllocatedState const& u, AllocatedState& v,
                                   UnaryOp unary_op) {
    for (auto i = 0; i < N; ++i) {
      v[i] = unary_op(u[i]);
    }
  }

  template <typename BinaryOp>
  friend void elementwise_binary_op(AllocatedState const& u, AllocatedState& v,
                                    BinaryOp binary_op) {
    for (auto i = 0; i < N; ++i) {
      binary_op(u[i], v[i]);
    }
  }

  template <typename BinaryOp>
  friend void elementwise_binary_op(AllocatedState const& u,
                                    AllocatedState const& v, AllocatedState& w,
                                    BinaryOp binary_op) {
    for (auto i = 0; i < N; ++i) {
      w[i] = binary_op(u[i], v[i]);
    }
  }

  friend auto inner_product(AllocatedState const& u, AllocatedState const& v) {
    auto sum = 0.0;
    for (auto i = 0; i < N; ++i) {
      sum += u[i] * v[i];
    }
    return sum;
  }

 private:
  std::unique_ptr<std::array<double, N>> state_{};
};
