#pragma once

#include <ranges>
#include <stdexcept>
#include <vector>

namespace vws = std::views;

/* Structure for output from ODE solver such as ODEIntegrator. */
class Output {
 public:
  static constexpr auto init_cap = 500;  // Initial capacity of storage arrays.

  /* Default constructor gives no output. */
  Output() = default;

  /* Constructor provides dense output at n_intervals_ equally spaced intervals.
   * If n_intervals_ <= 0, output is saved only at the actual integration steps.
   */
  explicit Output(int n_intervals)
      : n_intervals_{n_intervals}, suppress_output_{false} {
    dense_ = n_intervals_ > 0;
    x_values_.reserve(init_cap);
  }

  /* Called by the ODEIntegrator constructor, which passes neqn, the number of
   * equations, xlo, the starting point of the integration, and xhi, the ending
   * point. */
  void init(int neqn, double xlo, double xhi) {
    if (suppress_output_) {
      return;
    }
    y_values_.resize(neqn);
    for (auto& y : y_values_) {
      y.reserve(init_cap);
    }
    if (dense_) {
      x1_ = xlo;
      x2_ = xhi;
      interval_width_ = (x2_ - x1_) / n_intervals_;
      next_x_ = x1_ + interval_width_;
    }
  }

  /* Invokes dense_out function of stepper routine to produce output at xout.
   * Normally called by out rather than directly. Assumes that xout is between
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
    if (suppress_output_) {
      return;
    }
    if (dense_ && !first_call) {
      stepper.prepare_dense(stepper.hdid);
      out(stepper);
      return;
    }
    first_call = false;
    for (auto&& [saved_i, y_i] : vws::zip(y_values_, stepper.yout)) {
      saved_i.push_back(y_i);
    }
    x_values_.push_back(stepper.x);
  }

  /* Typically called by ODEIntegrator to produce dense output. Input variables
   * are the current value of x, the stepper, and the stepsize h. The routine
   * checks whether x is greater than the desired output point next_x_. If so,
   * it calls save_dense. */
  template <typename Stepper>
  void out(Stepper const& stepper) {
    if (!dense_) {
      throw std::runtime_error("dense output not set in Output");
    }
    while ((stepper.x - next_x_) * (x2_ - x1_) > 0.0) {
      save_dense(stepper, next_x_, stepper.hdid);
      next_x_ += interval_width_;
    }
  }

  /* Returns whether output is suppressed. */
  auto output_suppressed() const { return suppress_output_; }

  /* Returns whether dense output is generated. */
  auto is_dense() const { return dense_; }

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
  int n_intervals_;    // Number of intervals to save at for dense output.
  bool dense_{false};  // true if dense output requested.
  bool suppress_output_{true};  // Default is no output.
  bool first_call{true};        // true if first call to save.
};