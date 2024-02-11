#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Output.hpp"

template <typename Stepper>
struct ODEIntegrator {
  static constexpr auto max_step = 50'000;
  double eps;
  int nok;
  int nbad;
  int nvar;
  double x1;
  double x2;
  double hmin;
  bool dense;
  std::vector<double> y;
  std::vector<double> dydx;
  std::vector<double>& ystart;
  Output& out;
  typename Stepper::Dtype& derivs;
  Stepper stepper;
  int nstp;
  double x;
  double h;

  ODEIntegrator(std::vector<double>& ystartt, const double xx1,
                const double xx2, const double atol, const double rtol,
                const double h1, const double hminn, Output& outt,
                typename Stepper::Dtype& derivss);

  void integrate();
};

template <typename Stepper>
ODEIntegrator<Stepper>::ODEIntegrator(std::vector<double>& ystartt,
                                      const double xx1, const double xx2,
                                      const double atol, const double rtol,
                                      const double h1, const double hminn,
                                      Output& outt,
                                      typename Stepper::Dtype& derivss)
    : nvar(ystartt.size()),
      y(nvar),
      dydx(nvar),
      ystart(ystartt),
      x(xx1),
      nok(0),
      nbad(0),
      x1(xx1),
      x2(xx2),
      hmin(hminn),
      dense(outt.dense),
      out(outt),
      derivs(derivss),
      stepper(y, dydx, x, atol, rtol, dense) {
  eps = std::numeric_limits<double>::epsilon();
  h = x2 - x1 > 0.0 ? fabs(h1) : -fabs(h1);
  y = ystart;
  out.init(stepper.neqn, x1, x2);
}

template <typename Stepper>
void ODEIntegrator<Stepper>::integrate() {
  derivs(x, y, dydx);
  if (dense) {
    out.out(-1, x, y, stepper, h);
  } else {
    out.save(x, y);
  }
  for (nstp = 0; nstp < max_step; ++nstp) {
    if ((x + h * 1.0001 - x2) * (x2 - x1) > 0.0) {
      h = x2 - x;
    }
    stepper.step(h, derivs);
    if (stepper.hdid == h) {
      ++nok;
    } else {
      ++nbad;
    }
    if (dense) {
      out.out(nstp, x, y, stepper, stepper.hdid);
    } else {
      out.save(x, y);
    }
    if ((x - x2) * (x2 - x1) >= 0.0) {
      ystart = y;
      if (out.kmax > 0 &&
          fabs(out.xsave[out.count - 1] - x2) > 100.0 * fabs(x2) * eps) {
        out.save(x, y);
      }
      return;
    }
    if (fabs(stepper.hnext) <= hmin) {
      throw std::runtime_error("Step size too small in ODEIntegrator");
    }
    h = stepper.hnext;
  }
  throw std::runtime_error("Too many steps in routine ODEIntegrator");
}