#pragma once

#include <array>
#include <cmath>
#include <memory>

template <int N>
class VectorState {
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

  auto& operator+=(VectorState const& vs) {
    for (auto i = 0; i < N; ++i) {
      (*state_)[i] += (*vs.state_)[i];
    }
    return *this;
  }

  auto& operator-=(VectorState const& vs) {
    for (auto i = 0; i < N; ++i) {
      (*state_)[i] -= (*vs.state_)[i];
    }
    return *this;
  }
  auto operator-(VectorState const& vs) {
    auto temp = *this;
    temp -= vs;
    return temp;
  }

  auto& operator*=(double s) {
    for (auto i = 0; i < N; ++i) {
      (*state_)[i] *= s;
    }
    return *this;
  }
  auto operator*(double s) {
    auto temp = *this;
    temp *= s;
    return temp;
  }

  auto& operator/=(double s) {
    auto recip = 1.0 / s;
    return *this *= recip;
  }
  auto operator/(double s) {
    auto temp = *this;
    temp /= s;
    return temp;
  }

  auto& operator*=(VectorState const& vs) {
    for (auto i = 0; i < N; ++i) {
      (*state_)[i] *= (*vs.state_)[i];
    }
    return *this;
  }
  auto operator*(VectorState const& vs) {
    auto temp = *this;
    temp *= vs;
    return temp;
  }

  auto& operator/=(VectorState const& vs) {
    for (auto i = 0; i < N; ++i) {
      (*state_)[i] /= (*vs.state_)[i];
    }
    return *this;
  }
  auto operator/(VectorState const& vs) {
    auto temp = *this;
    temp /= vs;
    return temp;
  }

  auto& operator[](int i) { return (*state_)[i]; }
  auto const& operator[](int i) const { return (*state_)[i]; }

  auto mag2() const {
    auto sum = 0.0;
    for (auto i = 0; i < N; ++i) {
      sum += (*state_)[i] * (*state_)[i];
    }
    return sum;
  }

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

template <int N>
void fill(VectorState<N>& v, double value) {
  elementwise_unary_op(v, [value](auto& x) { x = value; });
}

template <int N>
void scalar_mult(VectorState<N>& v, double s) {
  elementwise_unary_op(v, [s](auto& x) { x *= s; });
}

template <int N>
void vector_add(VectorState<N> const& u, VectorState<N>& v) {
  elementwise_binary_op(u, v, [](auto a, auto& b) { b += a; });
}

template <int N>
void elementwise_mult(VectorState<N> const& u, VectorState<N>& v) {
  elementwise_binary_op(u, v, [](auto a, auto& b) { b *= a; });
}

template <int N>
void elementwise_mult_add(double a, VectorState<N> const& u,
                          VectorState<N>& v) {
  elementwise_binary_op(u, v, [a](auto b, auto& c) { c += a * b; });
}

template <int N>
void elementwise_mult_add(double a, VectorState<N> const& u,
                          VectorState<N> const& v, VectorState<N>& w) {
  elementwise_binary_op(u, v, w, [a](auto b, auto c) { return a * b + c; });
}

template <int N>
auto mult_ew(VectorState<N> const& u, VectorState<N> const& v) {
  auto temp = u;
  for (auto i = 0; i < N; ++i) {
    temp[i] *= v[i];
  }
  return temp;
}

template <int N>
auto max_ew(VectorState<N> const& u, VectorState<N> const& v) {
  auto temp = u;
  for (auto i = 0; i < N; ++i) {
    temp[i] = std::max(temp[i], v[i]);
  }
  return temp;
}

template <int N>
auto abs_ew(VectorState<N> const& u) {
  auto temp = u;
  for (auto i = 0; i < N; ++i) {
    temp[i] = std::abs(temp[i]);
  }
  return temp;
}
