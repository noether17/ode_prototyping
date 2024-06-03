#pragma once

#include <array>
#include <cmath>
#include <memory>

template <typename Derived>
struct ODEState {
  friend void fill(Derived& v, double value) {
    elementwise_unary_op(v, [value](auto& x) { x = value; });
  }

  friend void scalar_mult(Derived& v, double s) {
    elementwise_unary_op(v, [s](auto& x) { x *= s; });
  }

  friend void vector_add(Derived const& u, Derived& v) {
    elementwise_binary_op(u, v, [](auto a, auto& b) { b += a; });
  }

  friend void elementwise_mult(Derived const& u, Derived& v) {
    elementwise_binary_op(u, v, [](auto a, auto& b) { b *= a; });
  }

  friend void elementwise_mult_add(double a, Derived const& u, Derived& v) {
    elementwise_binary_op(u, v, [a](auto b, auto& c) { c += a * b; });
  }

  friend void elementwise_mult_add(double a, Derived const& u, Derived const& v,
                                   Derived& w) {
    elementwise_binary_op(u, v, w, [a](auto b, auto c) { return a * b + c; });
  }
};

template <int N>
class VectorState : public ODEState<VectorState<N>> {
 public:
  VectorState() : state_{std::make_unique<std::array<double, N>>()} {}
  VectorState(std::array<double, N> const& state)
      : state_{std::make_unique<std::array<double, N>>(state)} {}
  VectorState(VectorState const& vs)
      : state_{std::make_unique<std::array<double, N>>(*vs.state_)} {}
  VectorState(VectorState&& vs) = default;
  auto& operator=(VectorState const& vs) {
    if (this != &vs) {
      state_ = std::make_unique<std::array<double, N>>(*vs.state_);
    }
    return *this;
  }
  auto& operator=(VectorState&& vs) {
    state_ = std::move(vs.state_);
    return *this;
  }
  ~VectorState() = default;

  auto& operator[](int i) { return (*state_)[i]; }
  auto const& operator[](int i) const { return (*state_)[i]; }

  auto static constexpr size() { return N; }

  template <typename UnaryOp>
  friend void elementwise_unary_op(VectorState& v, UnaryOp unary_op) {
    for (auto i = 0; i < N; ++i) {
      unary_op(v[i]);
    }
  }

  template <typename UnaryOp>
  friend void elementwise_unary_op(VectorState const& u, VectorState& v,
                                   UnaryOp unary_op) {
    for (auto i = 0; i < N; ++i) {
      v[i] = unary_op(u[i]);
    }
  }

  template <typename BinaryOp>
  friend void elementwise_binary_op(VectorState const& u, VectorState& v,
                                    BinaryOp binary_op) {
    for (auto i = 0; i < N; ++i) {
      binary_op(u[i], v[i]);
    }
  }

  template <typename BinaryOp>
  friend void elementwise_binary_op(VectorState const& u, VectorState const& v,
                                    VectorState& w, BinaryOp binary_op) {
    for (auto i = 0; i < N; ++i) {
      w[i] = binary_op(u[i], v[i]);
    }
  }

  friend auto inner_product(VectorState const& u, VectorState const& v) {
    auto sum = 0.0;
    for (auto i = 0; i < N; ++i) {
      sum += u[i] * v[i];
    }
    return sum;
  }

 private:
  std::unique_ptr<std::array<double, N>> state_{};
};
