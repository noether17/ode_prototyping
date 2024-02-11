#pragma once

#include <stdexcept>
#include <vector>

/* Structure for output from ODE solver such as ODEIntegrator. */
struct Output {
  int kmax;  // Current capacity of storage arrays.
  int nvar;
  int nsave;   // Number of intervals to save at for dense output.
  bool dense;  // true if dense output requested.
  int count;   // Number of values actually saved.
  double x1;
  double x2;
  double xout;
  double dxout;
  std::vector<double> xsave;  // Results stored in the vector xsave[0...count-1]
  std::vector<std::vector<double>>
      ysave;  // and the matrix ysave[0...nvar-1][0...count-1].

  /* Default constructor gives no output. */
  Output() : kmax(-1), dense(false), count(0) {}

  /* Constructor provides dense output at nsave equally spaced intervals. If
   * nsave <= 0, output is saved only at the actual integration steps. */
  Output(const int nsavee) : kmax(500), nsave(nsavee), count(0), xsave(kmax) {
    dense = nsave > 0;
  }

  /* Called by the ODEIntegrator constructor, which passes neqn, the number of
   * equations, xlo, the starting point of the integration, and xhi, the ending
   * point. */
  void init(const int neqn, const double xlo, const double xhi) {
    nvar = neqn;
    if (kmax == -1) {
      return;
    }
    ysave.resize(nvar);
    for (auto& y : ysave) {
      y.resize(kmax);
    }
    if (dense) {
      x1 = xlo;
      x2 = xhi;
      xout = x1;
      dxout = (x2 - x1) / nsave;
    }
  }

  /* Resize storage arrays by a factor of two, keeping saved data. */
  void resize() {
    int kold = kmax;
    kmax *= 2;
    std::vector<double> tempvec(xsave);
    xsave.resize(kmax);
    for (int k = 0; k < kold; ++k) {
      xsave[k] = tempvec[k];
    }
    std::vector<std::vector<double>> tempmat(ysave);
    ysave.resize(nvar);
    for (int i = 0; i < nvar; ++i) {
      ysave[i].resize(kmax);
      for (int k = 0; k < kold; ++k) {
        ysave[i][k] = tempmat[i][k];
      }
    }
  }

  /* Invokes dense_out function of stepper routine to produce output at xout.
   * Normally called by out rather than directly. Assumes that xout is between
   * xold and xold + h, where the stepper must keep track of xold, the location
   * of the previous step, and x = xold + h, the current step. */
  template <typename Stepper>
  void save_dense(Stepper& stepper, const double xout, const double h) {
    if (count == kmax) {
      resize();
    }
    for (int i = 0; i < nvar; ++i) {
      ysave[i][count] = stepper.dense_out(i, xout, h);
    }
    xsave[count++] = xout;
  }

  /* Saves values of current x and y. */
  void save(const double x, std::vector<double>& y) {
    if (kmax <= 0) {
      return;
    }
    if (count == kmax) {
      resize();
    }
    for (int i = 0; i < nvar; ++i) {
      ysave[i][count] = y[i];
    }
    xsave[count++] = x;
  }

  /* Typically called by ODEIntegrator to produce dense output. Input variables
   * are nstp, the current step number, the current values of x and y, the
   * stepper, and the stepsize h. A call with nstp = -1 saves the initial
   * values. The routine checks whether x is greater than the desired output
   * point xout. If so, it calls save_dense. */
  template <typename Stepper>
  void out(const int nstp, const double x, std::vector<double>& y,
           Stepper& stepper, const double h) {
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
};