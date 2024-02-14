#pragma once

#include <vector>

/* Base class for all ODE algorithms. */
struct StepperBase {
  Output& out;
  double& x;
  double xold;  // Used for dense output.
  std::vector<double>& y;
  std::vector<double>& dydx;
  double atol;
  double rtol;
  double hdid;   // Actual stepsize accomplished by the step routine.
  double hnext;  // Step size predicted by the controller for the next step.
  double eps;
  int n;
  int neqn;                  // neqn = n except for StepperStoerm.
  std::vector<double> yout;  // New value of y and error estimate.
  std::vector<double> yerr;

  /* Input to the constructor are the dependent variable vector y[0...n-1] and
   * its derivative dydx[0...n-1] at the starting value of the independent
   * variable x. Also input are the absolute and relative tolerances, atol and
   * rtol, and the boolean dense, which is true if dense output is required. */
  StepperBase(std::vector<double>& yy, std::vector<double>& dydxx, double& xx,
              const double atoll, const double rtoll, Output& outt)
      : out{outt},
        x(xx),
        y(yy),
        dydx(dydxx),
        atol(atoll),
        rtol(rtoll),
        n(y.size()),
        neqn(n),
        yout(n),
        yerr(n) {}
};