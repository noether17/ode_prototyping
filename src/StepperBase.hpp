#pragma once

#include <limits>
#include <vector>

/* Base class for all ODE algorithms. */
struct StepperBase {
  auto static constexpr eps = std::numeric_limits<double>::epsilon();

  double& x;
  std::vector<double>& y;
  std::vector<double>& dydx;
  double atol;
  double rtol;
  double hdid;   // Actual stepsize accomplished by the step routine.
  double hnext;  // Step size predicted by the controller for the next step.
  int n;
  int neqn;                  // neqn = n except for StepperStoerm.
  std::vector<double> yout;  // New value of y and error estimate.
  std::vector<double> yerr;

  /* Input to the constructor are the dependent variable vector y[0...n-1] and
   * its derivative dydx[0...n-1] at the starting value of the independent
   * variable x. Also input are the absolute and relative tolerances, atol and
   * rtol. */
  StepperBase(std::vector<double>& yy, std::vector<double>& dydxx, double& xx,
              const double atoll, const double rtoll)
      : x(xx),
        y(yy),
        dydx(dydxx),
        atol(atoll),
        rtol(rtoll),
        n(y.size()),
        neqn(n),
        yout{yy},
        yerr(n) {}
};
