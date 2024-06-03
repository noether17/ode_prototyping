#pragma once

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
