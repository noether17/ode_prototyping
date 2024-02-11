#pragma once

#include <stdexcept>
#include <vector>

/* Structure for output from ODE solver such as ODEIntegrator. */
class Output {
  int nvar;
  int nsave;   // Number of intervals to save at for dense output.
  bool dense;  // true if dense output requested.
  bool suppress_output;
  double x1;
  double x2;
  double xout;
  double dxout;
  std::vector<double> xsave{};  // Results stored in the vector xsave
  std::vector<std::vector<double>>
      ysave{};  // and the matrix ysave[0...nvar-1].

 public:
  static constexpr auto init_cap = 500;  // Initial capacity of storage arrays.
  /* Default constructor gives no output. */
  Output() : suppress_output{true}, dense(false) {}

  /* Constructor provides dense output at nsave equally spaced intervals. If
   * nsave <= 0, output is saved only at the actual integration steps. */
  explicit Output(int nsavee) : suppress_output{false}, nsave(nsavee) {
    dense = nsave > 0;
    xsave.reserve(init_cap);
  }

  /* Called by the ODEIntegrator constructor, which passes neqn, the number of
   * equations, xlo, the starting point of the integration, and xhi, the ending
   * point. */
  void init(int neqn, double xlo, double xhi) {
    nvar = neqn;
    if (suppress_output) {
      return;
    }
    ysave.resize(nvar);
    for (auto& y : ysave) {
      y.reserve(init_cap);
    }
    if (dense) {
      x1 = xlo;
      x2 = xhi;
      xout = x1;
      dxout = (x2 - x1) / nsave;
    }
  }

  /* Invokes dense_out function of stepper routine to produce output at xout.
   * Normally called by out rather than directly. Assumes that xout is between
   * xold and xold + h, where the stepper must keep track of xold, the location
   * of the previous step, and x = xold + h, the current step. */
  template <typename Stepper>
  void save_dense(Stepper const& stepper, double xout, double h) {
    for (int i = 0; i < nvar; ++i) {
      ysave[i].push_back(stepper.dense_out(i, xout, h));
    }
    xsave.push_back(xout);
  }

  /* Saves values of current x and y. */
  void save(double x, std::vector<double> const& y) {
    if (suppress_output) {
      return;
    }
    for (int i = 0; i < nvar; ++i) {
      ysave[i].push_back(y[i]);
    }
    xsave.push_back(x);
  }

  /* Typically called by ODEIntegrator to produce dense output. Input variables
   * are nstp, the current step number, the current values of x and y, the
   * stepper, and the stepsize h. A call with nstp = -1 saves the initial
   * values. The routine checks whether x is greater than the desired output
   * point xout. If so, it calls save_dense. */
  template <typename Stepper>
  void out(int nstp, double x, std::vector<double> const& y,
           Stepper const& stepper, double h) {
    if (!dense) {
      throw std::runtime_error("dense output not set in Output");
    }
    if (nstp == -1) {
      save(x, y);
      xout += dxout;
    } else {
      while ((x - xout) * (x2 - x1) > 0.0) {
        save_dense(stepper, xout, h);
        xout += dxout;
      }
    }
  }

  /* Returns whether output is suppressed. */
  auto output_suppressed() const { return suppress_output; }

  /* Returns whether dense output is generated. */
  auto is_dense() const { return dense; }

  /* Returns the number of steps taken. */
  auto n_steps() const { return xsave.size(); }

  /* Returns the saved independent variable values. */
  auto const& x_values() const { return xsave; }

  /* Returns the saved dependent variable values. */
  auto const& y_values() const { return ysave; }
};