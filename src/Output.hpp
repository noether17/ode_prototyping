#pragma once

#include <ranges>
#include <stdexcept>
#include <vector>

namespace vws = std::views;

/* Policy class for providing no output of intermediate values. */
class NoOutput {
 public:
  void init(int, double, double) {}

  template <typename Stepper>
  void save(Stepper const&) {}
};  // TODO: Prevent instantiation of policy class outside of host by making
    // dtor protected.

/* Policy class for providing output at the actual integration steps. */
class RawOutput {
 public:
  static auto constexpr init_cap = 500;  // Initial capacity of storage arrays.

  void init(int neqn, double, double) {
    y_values_.resize(neqn);
    for (auto& y : y_values_) {
      y.reserve(init_cap);
    }
    x_values_.reserve(init_cap);
  }

  template <typename Stepper>
  void save(Stepper& stepper) {
    for (auto&& [saved_i, y_i] : vws::zip(y_values_, stepper.yout)) {
      saved_i.push_back(y_i);
    }
    x_values_.push_back(stepper.x);
  }

  auto n_steps() const { return x_values_.size(); }
  auto const& x_values() const { return x_values_; }
  auto const& y_values() const { return y_values_; }

 private:
  std::vector<double> x_values_{};  // Results stored in the vector x_values_
  std::vector<std::vector<double>> y_values_{};  // and the matrix y_values_.
};

/* Policy class for providing dense output. */
class DenseOutput {
 public:
  static auto constexpr init_cap = 500;  // Initial capacity of storage arrays.

  /* Constructor provides dense output at n_intervals_ equally spaced intervals.
   */
  explicit DenseOutput(int n_intervals) : n_intervals_{n_intervals} {}

  /* Called by the ODEIntegrator constructor, which passes neqn, the number of
   * equations, xlo, the starting point of the integration, and xhi, the ending
   * point. */
  void init(int neqn, double xlo, double xhi) {
    y_values_.resize(neqn);
    for (auto& y : y_values_) {
      y.reserve(init_cap);
    }
    x_values_.reserve(init_cap);
    x1_ = xlo;
    x2_ = xhi;
    interval_width_ = (x2_ - x1_) / n_intervals_;
    next_x_ = x1_ + interval_width_;
  }

  /* Invokes dense_out function of stepper routine to produce output at xout.
   * Normally called by save rather than directly. Assumes that xout is between
   * xold and xold + h, where the stepper must keep track of xold, the location
   * of the previous step, and x = xold + h, the current step. */
  template <typename Stepper>
  void save_dense(Stepper const& stepper, double xout, double h) {
    for (auto&& [i, y_i] : y_values_ | vws::enumerate) {
      y_i.push_back(stepper.dense_out(i, xout, h));
    }
    x_values_.push_back(xout);
  }

  /* Saves values of current x and y. */
  template <typename Stepper>
  void save(Stepper& stepper) {
    if (first_call) {
      first_call = false;
      for (auto&& [saved_i, y_i] : vws::zip(y_values_, stepper.yout)) {
        saved_i.push_back(y_i);
      }
      x_values_.push_back(stepper.x);
      return;
    }
    stepper.prepare_dense(stepper.hdid);
    while ((stepper.x - next_x_) * (x2_ - x1_) > 0.0) {
      save_dense(stepper, next_x_, stepper.hdid);
      next_x_ += interval_width_;
    }
  }

  /* Returns the number of steps taken. */
  auto n_steps() const { return x_values_.size(); }

  /* Returns the saved independent variable values. */
  auto const& x_values() const { return x_values_; }

  /* Returns the saved dependent variable values. */
  auto const& y_values() const { return y_values_; }

 private:
  std::vector<double> x_values_{};  // Results stored in the vector x_values_
  std::vector<std::vector<double>> y_values_{};  // and the matrix y_values_.
  double x1_;
  double x2_;
  double next_x_;
  double interval_width_;
  int n_intervals_;       // Number of intervals to save at for dense output.
  bool first_call{true};  // true if first call to save.
};