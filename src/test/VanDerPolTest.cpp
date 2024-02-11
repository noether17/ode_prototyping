#include <gtest/gtest.h>

#include <array>
#include <ranges>

#include "ODEIntegrator.hpp"
#include "StepperDopr5.hpp"

namespace vws = std::views;

class VanDerPolTest : public testing::Test {
 protected:
  static constexpr auto nvar = 2;
  static constexpr auto atol = 1.0e-3;
  static constexpr auto rtol = atol;
  static constexpr auto h1 = 0.01;
  static constexpr auto hmin = 0.0;
  static constexpr auto x1 = 0.0;
  static constexpr auto x2 = 2.0;
  static constexpr auto eps = 1.0e-3;
  static constexpr auto rhs_van = [](double x, std::vector<double> const& y,
                                     std::vector<double>& dydx) {
    dydx[0] = y[1];
    dydx[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps;
  };

  std::vector<double> ystart{2.0, 0.0};
};

TEST_F(VanDerPolTest, ActualIntegrationStepsAreConsistent) {
  auto out = Output(-1);  // -1 for actual integration steps.
  auto ode = ODEIntegrator<StepperDopr5<decltype(rhs_van)>>(
      ystart, x1, x2, atol, rtol, h1, hmin, out, rhs_van);

  ode.integrate();

  EXPECT_EQ(1177, out.count);

  EXPECT_DOUBLE_EQ(0.0, out.xsave[0]);
  EXPECT_DOUBLE_EQ(0.98436957642198519, out.xsave[out.count / 2]);
  EXPECT_DOUBLE_EQ(2.0, out.xsave[out.count - 1]);

  EXPECT_DOUBLE_EQ(2.0, out.ysave[0][0]);
  EXPECT_DOUBLE_EQ(-1.9010597230846302, out.ysave[0][out.count / 2]);
  EXPECT_DOUBLE_EQ(1.7644320190605265, out.ysave[0][out.count - 1]);

  EXPECT_DOUBLE_EQ(0.0, out.ysave[1][0]);
  EXPECT_DOUBLE_EQ(0.72447110662737324, out.ysave[1][out.count / 2]);
  EXPECT_DOUBLE_EQ(-0.83427005245677999, out.ysave[1][out.count - 1]);

  EXPECT_DOUBLE_EQ(1.7644320190605265, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.83427005245677999, ystart[1]);
}

TEST_F(VanDerPolTest, DenseOutputMatchesPython) {
  auto out = Output(20);
  auto ode = ODEIntegrator<StepperDopr5<decltype(rhs_van)>>(
      ystart, x1, x2, atol, rtol, h1, hmin, out, rhs_van);

  ode.integrate();

  // Reference values generated using scipy.integrate.solve_ivp().
  constexpr auto reference_x_values = std::array{0.0,
                                                 0.1,
                                                 0.2,
                                                 0.30000000000000004,
                                                 0.4,
                                                 0.5,
                                                 0.6,
                                                 0.7,
                                                 0.7999999999999999,
                                                 0.8999999999999999,
                                                 0.9999999999999999,
                                                 1.0999999999999999,
                                                 1.2,
                                                 1.3,
                                                 1.4000000000000001,
                                                 1.5000000000000002,
                                                 1.6000000000000003,
                                                 1.7000000000000004,
                                                 1.8000000000000005,
                                                 1.9000000000000006,
                                                 2.0};
  constexpr auto reference_y0_values = std::array{2.0,
                                                  1.9316114668840196,
                                                  1.8584936578161697,
                                                  1.779738838649533,
                                                  1.6936304673029232,
                                                  1.5973233455330436,
                                                  1.4853988650659193,
                                                  1.3445145377549454,
                                                  1.1027432107238924,
                                                  -1.9607574926871358,
                                                  -1.889636370708954,
                                                  -1.813401592162303,
                                                  -1.730627659606652,
                                                  -1.639036082901385,
                                                  -1.534583204376312,
                                                  -1.4085081361939422,
                                                  -1.2316250330343574,
                                                  1.9878333637317933,
                                                  1.9184567233175167,
                                                  1.8443946860836433,
                                                  1.7644320186384204};
  constexpr auto reference_y1_values = std::array{0.0,
                                                  -0.70723585006735,
                                                  -0.7569013815134848,
                                                  -0.8214089752926855,
                                                  -0.9065846314294861,
                                                  -1.0280880445964888,
                                                  -1.2278827867432427,
                                                  -1.6569111260368745,
                                                  -4.48537067895128,
                                                  0.6896276917781045,
                                                  0.7363385787399728,
                                                  0.79157138516551,
                                                  0.868805850565066,
                                                  0.9688135932292358,
                                                  1.131427622338704,
                                                  1.4281071047886105,
                                                  2.339031986623704,
                                                  -0.6741234554050665,
                                                  -0.7162074462396393,
                                                  -0.7682889897929661,
                                                  -0.8342700528712028};
  constexpr auto comp_tol =
      1.0e-9;  // Stricter accuracy requirement for comparison to Python
               // function that should be performing similar calculation.
  for (const auto& [ref_x, x] : vws::zip(reference_x_values, out.xsave)) {
    EXPECT_DOUBLE_EQ(ref_x, x);  // Independent variable should compare equal.
  }
  for (const auto& [ref_y0, y0] : vws::zip(reference_y0_values, out.ysave[0])) {
    EXPECT_NEAR(ref_y0, y0, comp_tol);
  }
  for (const auto& [ref_y1, y1] : vws::zip(reference_y1_values, out.ysave[1])) {
    EXPECT_NEAR(ref_y1, y1, comp_tol);
  }
  EXPECT_NEAR(1.7644320186384204, ystart[0], comp_tol);
  EXPECT_NEAR(-0.8342700528712028, ystart[1], comp_tol);
}

TEST_F(VanDerPolTest, DenseOutputIsConsistent) {
  auto out = Output(20);
  auto ode = ODEIntegrator<StepperDopr5<decltype(rhs_van)>>(
      ystart, x1, x2, atol, rtol, h1, hmin, out, rhs_van);

  ode.integrate();

  // Reference values from initial run to test for consistency.
  constexpr auto reference_x_values = std::array{0.0,
                                                 0.1,
                                                 0.2,
                                                 0.30000000000000004,
                                                 0.4,
                                                 0.5,
                                                 0.6,
                                                 0.7,
                                                 0.7999999999999999,
                                                 0.8999999999999999,
                                                 0.9999999999999999,
                                                 1.0999999999999999,
                                                 1.2,
                                                 1.3,
                                                 1.4000000000000001,
                                                 1.5000000000000002,
                                                 1.6000000000000003,
                                                 1.7000000000000004,
                                                 1.8000000000000005,
                                                 1.9000000000000006,
                                                 2.0};
  constexpr auto reference_y0_values = std::array{2.0,
                                                  1.9316114668840196,
                                                  1.8584936578161697,
                                                  1.779738838649533,
                                                  1.6936304673029232,
                                                  1.5973233455330451,
                                                  1.4853988650659193,
                                                  1.3445145377549466,
                                                  1.1027432107238968,
                                                  -1.9607574926907294,
                                                  -1.8896363707127886,
                                                  -1.8134015921664299,
                                                  -1.7306276596111723,
                                                  -1.6390360829064694,
                                                  -1.5345832043822263,
                                                  -1.4085081362013523,
                                                  -1.231625033046776,
                                                  1.9878333640723733,
                                                  1.9184567236794956,
                                                  1.844394686471839,
                                                  1.7644320190605265};
  constexpr auto reference_y1_values = std::array{0.0,
                                                  -0.70723585006735068,
                                                  -0.75690138151347441,
                                                  -0.82140897529267398,
                                                  -0.90658463142947066,
                                                  -1.0280880445966889,
                                                  -1.2278827867432773,
                                                  -1.6569111260367424,
                                                  -4.4853706789511456,
                                                  0.68962769177543404,
                                                  0.73633857873973074,
                                                  0.79157138514815228,
                                                  0.86880585055426407,
                                                  0.96881359325147665,
                                                  1.1314276223440602,
                                                  1.428107104729363,
                                                  2.3390319866124534,
                                                  -0.67412345520940009,
                                                  -0.71620744619303756,
                                                  -0.7682889892922099,
                                                  -0.83427005245677999};
  for (const auto& [ref_x, x] : vws::zip(reference_x_values, out.xsave)) {
    EXPECT_DOUBLE_EQ(ref_x, x);
  }
  for (const auto& [ref_y0, y0] : vws::zip(reference_y0_values, out.ysave[0])) {
    EXPECT_DOUBLE_EQ(ref_y0, y0);
  }
  for (const auto& [ref_y1, y1] : vws::zip(reference_y1_values, out.ysave[1])) {
    EXPECT_DOUBLE_EQ(ref_y1, y1);
  }
  EXPECT_DOUBLE_EQ(1.7644320190605265, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.83427005245677999, ystart[1]);
}