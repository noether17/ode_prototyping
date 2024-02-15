#pragma once

#include <cmath>
#include <vector>

#include "StepperBase.hpp"

/* Dormand-Prince fifth-order Runge-Kutta step with monitoring of local
 * truncation error to ensure accuracy and adjust stepsize. */
template <typename D>
struct StepperDopr5 : StepperBase {
  typedef D Dtype;  // Make the type of derivs available to ODEIntegrator.
  std::vector<double> k2;
  std::vector<double> k3;
  std::vector<double> k4;
  std::vector<double> k5;
  std::vector<double> k6;
  std::vector<double> rcont1;
  std::vector<double> rcont2;
  std::vector<double> rcont3;
  std::vector<double> rcont4;
  std::vector<double> rcont5;
  std::vector<double> dydxnew;
  bool first_step{true};

  StepperDopr5(std::vector<double>& yy, std::vector<double>& dydxx, double& xx,
               const double atoll, const double rtoll, Output& outt);

  void step(const double htry, D& derivs);
  void save(D& derivs);
  void dy(const double h, D& derivs);
  void prepare_dense(const double h, D& derivs);
  double dense_out(const int i, const double x, const double h) const;
  double error();
  struct Controller {
    double hnext;
    double errold;
    bool reject;

    Controller();
    bool success(const double err, double& h);
  };
  Controller con;
};

/* Input to the constructor are the dependent variable y[0...n-1] and its
 * derivative dydx[0...n-1] at the starting value of the independent variable x.
 * Also input are the absolute and relative tolerances, atol and rtol, and the
 * boolean dense, which is true if dense output is required. */
template <typename D>
StepperDopr5<D>::StepperDopr5(std::vector<double>& yy,
                              std::vector<double>& dydxx, double& xx,
                              const double atoll, const double rtoll,
                              Output& outt)
    : StepperBase(yy, dydxx, xx, atoll, rtoll, outt),
      k2(n),
      k3(n),
      k4(n),
      k5(n),
      k6(n),
      rcont1(n),
      rcont2(n),
      rcont3(n),
      rcont4(n),
      rcont5(n),
      dydxnew(n) {
  eps = std::numeric_limits<double>::epsilon();
}

/* Attempts a step with stepsize htry. On output, y and x are replaced by their
 * new values, hdid is the stepsize that was actually accomplished, and hnext is
 * the estimated next stepsize. */
template <typename D>
void StepperDopr5<D>::step(const double htry, D& derivs) {
  double h = htry;  // Set stepsize to the initial trial value.
  for (;;) {
    dy(h, derivs);               // Take a step.
    const double err = error();  // Evaluate accuracy.
    if (con.success(err, h)) {
      break;
    }  // Step rejected. Try again with reduced h set by controller.
    if (fabs(h) <= fabs(x) * eps) {
      throw std::runtime_error("step size underflow in StepperDopr5");
    }
  }
  xold = x;  // Used for dense output.
  x += (hdid = h);
  save(derivs);
  dydx = dydxnew;  // Reuse last derivative evaluation for next step.
  y = yout;
  hnext = con.hnext;
}

template <typename D>
void StepperDopr5<D>::save(D& derivs) {
  if (out.is_dense() && !first_step) {
    prepare_dense(hdid, derivs);
    out.out(*this);
  } else {
    out.save(*this);
    first_step = false;
  }
}

/* Given values for n variables y[0...n-1] and their derivatives dydx[0...n-1]
 * known at x, use the fifth-order Dormand-Prince Runge-Kutta method to advance
 * the solution over an interval h and store the incremented variables in
 * yout[0...n-1]. Also store an estimate of the local truncation error in yerr
 * using the embedded fourth-order method. */
template <typename D>
void StepperDopr5<D>::dy(const double h, D& derivs) {
  static auto constexpr c2 = 0.2;
  static auto constexpr c3 = 0.3;
  static auto constexpr c4 = 0.8;
  static auto constexpr c5 = 8.0 / 9.0;
  static auto constexpr a21 = 0.2;
  static auto constexpr a31 = 3.0 / 40.0;
  static auto constexpr a32 = 9.0 / 40.0;
  static auto constexpr a41 = 44.0 / 45.0;
  static auto constexpr a42 = -56.0 / 15.0;
  static auto constexpr a43 = 32.0 / 9.0;
  static auto constexpr a51 = 19372.0 / 6561.0;
  static auto constexpr a52 = -25360.0 / 2187.0;
  static auto constexpr a53 = 64448.0 / 6561.0;
  static auto constexpr a54 = -212.0 / 729.0;
  static auto constexpr a61 = 9017.0 / 3168.0;
  static auto constexpr a62 = -355.0 / 33.0;
  static auto constexpr a63 = 46732.0 / 5247.0;
  static auto constexpr a64 = 49.0 / 176.0;
  static auto constexpr a65 = -5103.0 / 18656.0;
  static auto constexpr a71 = 35.0 / 384.0;
  static auto constexpr a73 = 500.0 / 1113.0;
  static auto constexpr a74 = 125.0 / 192.0;
  static auto constexpr a75 = -2187.0 / 6784.0;
  static auto constexpr a76 = 11.0 / 84.0;
  static auto constexpr e1 = 71.0 / 57600.0;
  static auto constexpr e3 = -71.0 / 16695.0;
  static auto constexpr e4 = 71.0 / 1920.0;
  static auto constexpr e5 = -17253.0 / 339200.0;
  static auto constexpr e6 = 22.0 / 525.0;
  static auto constexpr e7 = -1.0 / 40.0;

  std::vector<double> ytemp(n);
  int i;
  for (i = 0; i < n; ++i) {  // First step.
    ytemp[i] = y[i] + h * a21 * dydx[i];
  }
  derivs(x + c2 * h, ytemp, k2);  // Second step.
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (a31 * dydx[i] + a32 * k2[i]);
  }
  derivs(x + c3 * h, ytemp, k3);  // Third step.
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (a41 * dydx[i] + a42 * k2[i] + a43 * k3[i]);
  }
  derivs(x + c4 * h, ytemp, k4);  // Fourth step.
  for (i = 0; i < n; ++i) {
    ytemp[i] =
        y[i] + h * (a51 * dydx[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
  }
  derivs(x + c5 * h, ytemp, k5);  // Fifth step.
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (a61 * dydx[i] + a62 * k2[i] + a63 * k3[i] +
                           a64 * k4[i] + a65 * k5[i]);
  }
  double xph = x + h;
  derivs(xph, ytemp, k6);    // Sixth step.
  for (i = 0; i < n; ++i) {  // Accumulate increments with proper weights.
    yout[i] = y[i] + h * (a71 * dydx[i] + a73 * k3[i] + a74 * k4[i] +
                          a75 * k5[i] + a76 * k6[i]);
  }
  derivs(xph, yout, dydxnew);  // Will also be first evaluation for next step.
  for (i = 0; i < n; ++i) {  // Estimate error as difference between fourth- and
                             // fifth-order methods.
    yerr[i] = h * (e1 * dydx[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] +
                   e6 * k6[i] + e7 * dydxnew[i]);
  }
}

/* Store coefficients of interpolating polynomial for dense output in
 * rcont1...rcont5. */
template <typename D>
void StepperDopr5<D>::prepare_dense(const double h, D&) {
  static auto constexpr d1 = -12715105075.0 / 11282082432.0;
  static auto constexpr d3 = 87487479700.0 / 32700410799.0;
  static auto constexpr d4 = -10690763975.0 / 1880347072.0;
  static auto constexpr d5 = 701980252875.0 / 199316789632.0;
  static auto constexpr d6 = -1453857185.0 / 822651844.0;
  static auto constexpr d7 = 69997945.0 / 29380423.0;
  for (int i = 0; i < n; ++i) {
    rcont1[i] = y[i];
    double ydiff = yout[i] - y[i];
    rcont2[i] = ydiff;
    double bspl = h * dydx[i] - ydiff;
    rcont3[i] = bspl;
    rcont4[i] = ydiff - h * dydxnew[i] - bspl;
    rcont5[i] = h * (d1 * dydx[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] +
                     d6 * k6[i] + d7 * dydxnew[i]);
  }
}

/* Evaluate interpolating polynomial for y[i] at location x, where xold <= x <=
 * xold + h. */
template <typename D>
double StepperDopr5<D>::dense_out(const int i, const double x,
                                  const double h) const {
  double s = (x - xold) / h;
  double s1 = 1.0 - s;
  return rcont1[i] +
         s * (rcont2[i] + s1 * (rcont3[i] + s * (rcont4[i] + s1 * rcont5[i])));
}

/* Use yerr to compute norm of scaled error estimate. A value less than one
 * means the step was successful. */
template <typename D>
double StepperDopr5<D>::error() {
  double err = 0.0;
  for (int i = 0; i < n; ++i) {
    double sk = atol + rtol * std::max(fabs(y[i]), fabs(yout[i]));
    err += pow(yerr[i] / sk, 2);
  }
  return sqrt(err / n);
}

/* Step size controller for fifth-order Dormand-Prince method. */
template <typename D>
StepperDopr5<D>::Controller::Controller() : errold(1.0e-4), reject(false) {}

/* Returns true if err <= 1, false otherwise. If step was successful, sets hnext
 * to the estimated optimal stepsize for the next step. If the step failed,
 * reduces h appropriately for another try. */
template <typename D>
bool StepperDopr5<D>::Controller::success(const double err, double& h) {
  static auto constexpr beta =
      0.0;  // Set beta to a nonzero value for PI control. beta = 0.04-0.08 is a
            // good default.
  static auto constexpr alpha = 0.2 - beta * 0.75;
  static auto constexpr safe = 0.9;
  static auto constexpr minscale = 0.2;
  static auto constexpr maxscale = 10.0;
  double scale;
  if (err <= 1.0) {  // Step succeeded. Compute hnext.
    if (err == 0.0) {
      scale = maxscale;
    } else {  // PI control if beta != 0.
      scale = safe * pow(err, -alpha) * pow(errold, beta);
      if (scale < minscale) {  // Ensure minscale <= hnext / h <= maxscale.
        scale = minscale;
      }
      if (scale > maxscale) {
        scale = maxscale;
      }
    }
    if (reject) {  // Don't let step increase if last one was rejected.
      hnext = h * std::min(scale, 1.0);
    } else {
      hnext = h * scale;
    }
    errold = std::max(err, 1.0e-4);  // Bookkeeping for next call.
    reject = false;
    return true;
  } else {  // Truncation error too large, reduce stepsize.
    scale = std::max(safe * pow(err, -alpha), minscale);
    h *= scale;
    reject = true;
    return false;
  }
}