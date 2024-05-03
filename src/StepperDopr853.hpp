#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "Dopr853_constants.hpp"
#include "StepperBase.hpp"

auto constexpr SQR(auto x) { return x * x; }

/* Specifies the additional data needed to produce dense output, to be held by
 * the DenseOutput policy class and modified by prepare_dense(). The precise
 * method of generating dense output varies by integration method, so each
 * integration method that is intended to be used with the DenseOutput policy
 * must provide prepare_dense() and dense_out() functions as well as a data
 * structure for these functions to operate on. If these are not provided, the
 * method may still be used with the NoOutput or RawOutput policies. */
struct Dopr853DenseData {
  std::vector<double> rcont1{};
  std::vector<double> rcont2{};
  std::vector<double> rcont3{};
  std::vector<double> rcont4{};
  std::vector<double> rcont5{};
  std::vector<double> rcont6{};
  std::vector<double> rcont7{};
  std::vector<double> rcont8{};
};

/* Dormand-Prince eighth-order Runge-Kutta step with monitoring of local
 * truncation error to ensure accuracy and adjust stepsize. Only important
 * differences from StepperDopr5 are commented. */
template <class D, typename OP>
struct StepperDopr853 : StepperBase, OP {
  using Dtype = D;
  using OutputPolicy = OP;
  std::vector<double> yerr2;  // Use a second error estimator.
  D& derivs;
  std::vector<double> k2{};
  std::vector<double> k3{};
  std::vector<double> k4{};
  std::vector<double> k5{};
  std::vector<double> k6{};
  std::vector<double> k7{};
  std::vector<double> k8{};
  std::vector<double> k9{};
  std::vector<double> k10{};
  std::vector<double> dydxnew;

  StepperDopr853(std::vector<double>& yy, std::vector<double>& dydxx,
                 double& xx, double atoll, double rtoll, D& derivss);

  void step(double& htry, D& derivs);
  void save();
  void dy(double h, D& derivs);
  void prepare_dense(double h, Dopr853DenseData& dense_data);
  double dense_out(int i, double x, double h,
                   Dopr853DenseData const& dense_data) const;
  double error(double h);
  struct Controller {
    double hnext{};
    double errold{1.0e-4};
    bool reject{false};

    bool success(double err, double& h);
  };
  Controller con;
};

template <class D, typename OP>
StepperDopr853<D, OP>::StepperDopr853(std::vector<double>& yy,
                                      std::vector<double>& dydxx, double& xx,
                                      double atoll, double rtoll, D& derivss)
    : StepperBase(yy, dydxx, xx, atoll, rtoll),
      OP{},
      yerr2(n),
      derivs{derivss},
      k2(n),
      k3(n),
      k4(n),
      k5(n),
      k6(n),
      k7(n),
      k8(n),
      k9(n),
      k10(n),
      dydxnew(n) {}

/* This routine is essentially the same as the one in StepperDopr5 except that
 * derivs is called here rather than in dy because this method does not use
 * FSAL. */
template <class D, typename OP>
void StepperDopr853<D, OP>::step(double& htry, D& derivs) {
  double h = htry;
  for (;;) {
    dy(h, derivs);
    auto const err = error(h);
    if (con.success(err, h)) {
      break;
    }
    if (std::abs(h) <= std::abs(x) * eps) {
      throw std::runtime_error("step size underflow in StepperDopr853");
    }
  }
  derivs(x + h, yout, dydxnew);
  x += (hdid = h);
  save();
  dydx = dydxnew;
  y = yout;
  htry = con.hnext;
}

template <typename D, typename OP>
void StepperDopr853<D, OP>::save() {
  OP::save(*this);
}

template <class D, typename OP>
void StepperDopr853<D, OP>::dy(double h, D& derivs) {
  std::vector<double> ytemp(n);
  int i;
  for (i = 0; i < n; ++i) {  // Twelve stages.
    ytemp[i] = y[i] + h * DP853::a21 * dydx[i];
  }
  derivs(x + DP853::c2 * h, ytemp, k2);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a31 * dydx[i] + DP853::a32 * k2[i]);
  }
  derivs(x + DP853::c3 * h, ytemp, k3);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a41 * dydx[i] + DP853::a43 * k3[i]);
  }
  derivs(x + DP853::c4 * h, ytemp, k4);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a51 * dydx[i] + DP853::a53 * k3[i] +
                           DP853::a54 * k4[i]);
  }
  derivs(x + DP853::c5 * h, ytemp, k5);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a61 * dydx[i] + DP853::a64 * k4[i] +
                           DP853::a65 * k5[i]);
  }
  derivs(x + DP853::c6 * h, ytemp, k6);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a71 * dydx[i] + DP853::a74 * k4[i] +
                           DP853::a75 * k5[i] + DP853::a76 * k6[i]);
  }
  derivs(x + DP853::c7 * h, ytemp, k7);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a81 * dydx[i] + DP853::a84 * k4[i] +
                           DP853::a85 * k5[i] + DP853::a86 * k6[i] +
                           DP853::a87 * k7[i]);
  }
  derivs(x + DP853::c8 * h, ytemp, k8);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a91 * dydx[i] + DP853::a94 * k4[i] +
                           DP853::a95 * k5[i] + DP853::a96 * k6[i] +
                           DP853::a97 * k7[i] + DP853::a98 * k8[i]);
  }
  derivs(x + DP853::c9 * h, ytemp, k9);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a101 * dydx[i] + DP853::a104 * k4[i] +
                           DP853::a105 * k5[i] + DP853::a106 * k6[i] +
                           DP853::a107 * k7[i] + DP853::a108 * k8[i] +
                           DP853::a109 * k9[i]);
  }
  derivs(x + DP853::c10 * h, ytemp, k10);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a111 * dydx[i] + DP853::a114 * k4[i] +
                           DP853::a115 * k5[i] + DP853::a116 * k6[i] +
                           DP853::a117 * k7[i] + DP853::a118 * k8[i] +
                           DP853::a119 * k9[i] + DP853::a1110 * k10[i]);
  }
  derivs(x + DP853::c11 * h, ytemp, k2);
  double xph = x + h;
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a121 * dydx[i] + DP853::a124 * k4[i] +
                           DP853::a125 * k5[i] + DP853::a126 * k6[i] +
                           DP853::a127 * k7[i] + DP853::a128 * k8[i] +
                           DP853::a129 * k9[i] + DP853::a1210 * k10[i] +
                           DP853::a1211 * k2[i]);
  }
  derivs(xph, ytemp, k3);
  for (i = 0; i < n; ++i) {
    k4[i] = DP853::b1 * dydx[i] + DP853::b6 * k6[i] + DP853::b7 * k7[i] +
            DP853::b8 * k8[i] + DP853::b9 * k9[i] + DP853::b10 * k10[i] +
            DP853::b11 * k2[i] + DP853::b12 * k3[i];
    yout[i] = y[i] + h * k4[i];
  }
  for (i = 0; i < n; ++i) {  // Two error estimators.
    yerr[i] = k4[i] - DP853::bhh1 * dydx[i] - DP853::bhh2 * k9[i] -
              DP853::bhh3 * k3[i];
    yerr2[i] = DP853::er1 * dydx[i] + DP853::er6 * k6[i] + DP853::er7 * k7[i] +
               DP853::er8 * k8[i] + DP853::er9 * k9[i] + DP853::er10 * k10[i] +
               DP853::er11 * k2[i] + DP853::er12 * k3[i];
  }
}

template <class D, typename OP>
void StepperDopr853<D, OP>::prepare_dense(double h,
                                          Dopr853DenseData& dense_data) {
  if (dense_data.rcont1.empty()) {
    dense_data.rcont1.resize(n);
    dense_data.rcont2.resize(n);
    dense_data.rcont3.resize(n);
    dense_data.rcont4.resize(n);
    dense_data.rcont5.resize(n);
    dense_data.rcont6.resize(n);
    dense_data.rcont7.resize(n);
    dense_data.rcont8.resize(n);
  }
  int i;
  double ydiff, bspl;
  std::vector<double> ytemp(n);
  for (i = 0; i < n; ++i) {
    dense_data.rcont1[i] = y[i];
    ydiff = yout[i] - y[i];
    dense_data.rcont2[i] = ydiff;
    bspl = h * dydx[i] - ydiff;
    dense_data.rcont3[i] = bspl;
    dense_data.rcont4[i] = ydiff - h * dydxnew[i] - bspl;
    dense_data.rcont5[i] = DP853::d41 * dydx[i] + DP853::d46 * k6[i] +
                           DP853::d47 * k7[i] + DP853::d48 * k8[i] +
                           DP853::d49 * k9[i] + DP853::d410 * k10[i] +
                           DP853::d411 * k2[i] + DP853::d412 * k3[i];
    dense_data.rcont6[i] = DP853::d51 * dydx[i] + DP853::d56 * k6[i] +
                           DP853::d57 * k7[i] + DP853::d58 * k8[i] +
                           DP853::d59 * k9[i] + DP853::d510 * k10[i] +
                           DP853::d511 * k2[i] + DP853::d512 * k3[i];
    dense_data.rcont7[i] = DP853::d61 * dydx[i] + DP853::d66 * k6[i] +
                           DP853::d67 * k7[i] + DP853::d68 * k8[i] +
                           DP853::d69 * k9[i] + DP853::d610 * k10[i] +
                           DP853::d611 * k2[i] + DP853::d612 * k3[i];
    dense_data.rcont8[i] = DP853::d71 * dydx[i] + DP853::d76 * k6[i] +
                           DP853::d77 * k7[i] + DP853::d78 * k8[i] +
                           DP853::d79 * k9[i] + DP853::d710 * k10[i] +
                           DP853::d711 * k2[i] + DP853::d712 * k3[i];
  }
  for (i = 0; i < n; ++i) {  // The three extra function evaluations.
    ytemp[i] = y[i] + h * (DP853::a141 * dydx[i] + DP853::a147 * k7[i] +
                           DP853::a148 * k8[i] + DP853::a149 * k9[i] +
                           DP853::a1410 * k10[i] + DP853::a1411 * k2[i] +
                           DP853::a1412 * k3[i] + DP853::a1413 * dydxnew[i]);
  }
  derivs(x + DP853::c14 * h, ytemp, k10);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a151 * dydx[i] + DP853::a156 * k6[i] +
                           DP853::a157 * k7[i] + DP853::a158 * k8[i] +
                           DP853::a1511 * k2[i] + DP853::a1512 * k3[i] +
                           DP853::a1513 * dydxnew[i] + DP853::a1514 * k10[i]);
  }
  derivs(x + DP853::c15 * h, ytemp, k2);
  for (i = 0; i < n; ++i) {
    ytemp[i] = y[i] + h * (DP853::a161 * dydx[i] + DP853::a166 * k6[i] +
                           DP853::a167 * k7[i] + DP853::a168 * k8[i] +
                           DP853::a169 * k9[i] + DP853::a1613 * dydxnew[i] +
                           DP853::a1614 * k10[i] + DP853::a1615 * k2[i]);
  }
  derivs(x + DP853::c16 * h, ytemp, k3);
  for (i = 0; i < n; ++i) {
    dense_data.rcont5[i] =
        h * (dense_data.rcont5[i] + DP853::d413 * dydxnew[i] +
             DP853::d414 * k10[i] + DP853::d415 * k2[i] + DP853::d416 * k3[i]);
    dense_data.rcont6[i] =
        h * (dense_data.rcont6[i] + DP853::d513 * dydxnew[i] +
             DP853::d514 * k10[i] + DP853::d515 * k2[i] + DP853::d516 * k3[i]);
    dense_data.rcont7[i] =
        h * (dense_data.rcont7[i] + DP853::d613 * dydxnew[i] +
             DP853::d614 * k10[i] + DP853::d615 * k2[i] + DP853::d616 * k3[i]);
    dense_data.rcont8[i] =
        h * (dense_data.rcont8[i] + DP853::d713 * dydxnew[i] +
             DP853::d714 * k10[i] + DP853::d715 * k2[i] + DP853::d716 * k3[i]);
  }
}

template <class D, typename OP>
double StepperDopr853<D, OP>::dense_out(
    int i, double x, double h, Dopr853DenseData const& dense_data) const {
  double s = (x - (StepperBase::x - h)) / h;
  double s1 = 1.0 - s;
  return dense_data.rcont1[i] +
         s * (dense_data.rcont2[i] +
              s1 * (dense_data.rcont3[i] +
                    s * (dense_data.rcont4[i] +
                         s1 * (dense_data.rcont5[i] +
                               s * (dense_data.rcont6[i] +
                                    s1 * (dense_data.rcont7[i] +
                                          s * dense_data.rcont8[i]))))));
}

template <class D, typename OP>
double StepperDopr853<D, OP>::error(double h) {
  auto err = 0.0;
  auto err2 = 0.0;
  for (int i = 0; i < n; ++i) {
    auto sk = atol + rtol * std::max(std::abs(y[i]), std::abs(yout[i]));
    err2 += SQR(yerr[i] / sk);
    err += SQR(yerr2[i] / sk);
  }
  auto deno = err + 0.01 * err2;
  if (deno <= 0.0) {
    deno = 1.0;
  }
  return std::abs(h) * err *
         sqrt(1.0 / (n * deno));  // The factor of h is here because it was
                                  // omitted when yerr and yerr2 were formed.
}

template <class D, typename OP>
bool StepperDopr853<D, OP>::Controller::success(double err, double& h) {
  /* Same controller as StepperDopr5 except different values of the constants.
   */
  static double constexpr beta = 0.0, alpha = 1.0 / 8.0 - beta * 0.2,
                          safe = 0.9, minscale = 0.333, maxscale = 6.0;
  double scale;
  if (err <= 1.0) {
    if (err == 0.0) {
      scale = maxscale;
    } else {
      scale = safe * pow(err, -alpha) * pow(errold, beta);
      if (scale < minscale) {
        scale = minscale;
      }
      if (scale > maxscale) {
        scale = maxscale;
      }
    }
    if (reject) {
      hnext = h * std::min(scale, 1.0);
    } else {
      hnext = h * scale;
    }
    errold = std::max(err, 1.0e-4);
    reject = false;
    return true;
  } else {
    scale = std::max(safe * pow(err, -alpha), minscale);
    h *= scale;
    reject = true;
    return false;
  }
}

/* Alias template for simplifying instantiation of StepperDopr853 with dense
 * output. */
template <typename D, template <typename> class OP>
using StepperDopr853Dense = StepperDopr853<D, OP<Dopr853DenseData>>;
