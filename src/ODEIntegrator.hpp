#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Output.hpp"

/* Driver for ODE solvers with adaptive stepsize control. The template parameter
 * should be one of the derived classes of StepperBase defining a particular
 * integration algorithm. */
template <typename Stepper>
struct ODEIntegrator {
  static constexpr auto max_step = 50'000;  // Take at most max_step steps.

  std::vector<double> y;
  std::vector<double> dydx;
  std::vector<double>& ystart;
  typename Stepper::Dtype& derivs;  // Get the type of derivs from the stepper.
  Stepper stepper;
  double eps;
  double x1;
  double x2;
  double hmin;
  double x;
  double h;
  int nok;
  int nbad;
  int nvar;
  int nstp;
  bool dense;  // true if dense output is requested by out.

  /* Constructor sets everything up. The routine integrates starting values
   * ystart[0...nvar-1] from xx1 to xx2 with absolute tolerance atol and
   * relative tolerance rtol. The quantity h1 should be set as a guessed first
   * stepsize, hmin as the minimum allowed stepsize (can be zero). An Output
   * object out should be input to control the saving of intermediate values. On
   * output, nok and nbad are the number of good and bad (but retried and fixed)
   * steps taken, and ystart is replaced by values at the end of the integration
   * interval. derivs is the user-supplied routine (function or functor) for
   * calculating the right-hand side derivative. */
  ODEIntegrator(std::vector<double>& ystartt, const double xx1,
                const double xx2, const double atol, const double rtol,
                const double h1, const double hminn, Output& outt,
                typename Stepper::Dtype& derivss);

  void integrate();  // Does the actual integration.
};

template <typename Stepper>
ODEIntegrator<Stepper>::ODEIntegrator(std::vector<double>& ystartt,
                                      const double xx1, const double xx2,
                                      const double atol, const double rtol,
                                      const double h1, const double hminn,
                                      Output& outt,
                                      typename Stepper::Dtype& derivss)
    : y(ystartt.size()),
      dydx(ystartt.size()),
      ystart(ystartt),
      derivs(derivss),
      stepper(y, dydx, x, atol, rtol, dense, outt),
      x1(xx1),
      x2(xx2),
      hmin(hminn),
      x(xx1),
      nok(0),
      nbad(0),
      nvar(ystartt.size()),
      dense(outt.is_dense()) {
  eps = std::numeric_limits<double>::epsilon();
  h = x2 - x1 > 0.0 ? fabs(h1) : -fabs(h1);
  y = ystart;
  stepper.out.init(stepper.neqn, x1, x2);
}

template <typename Stepper>
void ODEIntegrator<Stepper>::integrate() {
  derivs(x, y, dydx);
  stepper.out.save(x, y);
  for (nstp = 0; nstp < max_step; ++nstp) {
    if ((x + h * 1.0001 - x2) * (x2 - x1) > 0.0) {
      h = x2 - x;  // If stepsize can overshoot, decrease.
    }
    stepper.step(h, derivs);  // Take a step.
    if (stepper.hdid == h) {
      ++nok;
    } else {
      ++nbad;
    }
    if (dense) {
      stepper.out.out(x, stepper, stepper.hdid);
    } else {
      stepper.out.save(x, y);
    }
    if ((x - x2) * (x2 - x1) >= 0.0) {  // Are we done?
      ystart = y;                       // Update ystart.
      return;                           // Normal exit.
    }
    if (fabs(stepper.hnext) <= hmin) {
      throw std::runtime_error("Step size too small in ODEIntegrator");
    }
    h = stepper.hnext;
  }
  throw std::runtime_error("Too many steps in routine ODEIntegrator");
}