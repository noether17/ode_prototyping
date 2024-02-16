#include <gtest/gtest.h>

#include <array>
#include <ranges>

#include "ODEIntegrator.hpp"
#include "StepperDopr5.hpp"

namespace vws = std::views;

class SimpleOscillatorTest : public testing::Test {
 protected:
  static auto constexpr nvar = 2;
  static auto constexpr atol = 1.0e-10;
  static auto constexpr rtol = atol;
  static auto constexpr h1 = 0.01;
  static auto constexpr hmin = 0.0;
  static auto constexpr x1 = 0.0;
  static auto constexpr x2 = 2.0;
  static auto constexpr rhs_osc = [](double, std::vector<double> const& y,
                                     std::vector<double>& dydx) {
    dydx[0] = y[1];
    dydx[1] = -y[0];
  };

  std::vector<double> ystart{1.0, 0.0};

  using Dopr5Integrator =
      ODEIntegrator<StepperDopr5<decltype(rhs_osc), Output>>;
  using Dopr5IntegratorRawOutput =
      ODEIntegrator<StepperDopr5<decltype(rhs_osc), RawOutput>>;
};

/* Consistency tests (testing for double equality) are to ensure no accidental
 * algorithm changes are made during refactoring. These tests are far stricter
 * than the actual requirements. If an intentional change in algorithm results
 * in small differences in output, these values may be updated. */
TEST_F(SimpleOscillatorTest, ActualIntegrationStepsAreConsistent) {
  auto ode = Dopr5IntegratorRawOutput(ystart, x1, x2, atol, rtol, h1, hmin,
                                      RawOutput{}, rhs_osc);
  auto const& out = ode.stepper;

  ode.integrate();

  EXPECT_EQ(49, out.n_steps());

  EXPECT_DOUBLE_EQ(0.0, out.x_values()[0]);
  EXPECT_DOUBLE_EQ(1.0004288786936162, out.x_values()[out.n_steps() / 2]);
  EXPECT_DOUBLE_EQ(2.0, out.x_values()[out.n_steps() - 1]);

  EXPECT_DOUBLE_EQ(1.0, out.y_values()[0][0]);
  EXPECT_DOUBLE_EQ(0.53994136718681807, out.y_values()[0][out.n_steps() / 2]);
  EXPECT_DOUBLE_EQ(-0.41614683651997603, out.y_values()[0][out.n_steps() - 1]);

  EXPECT_DOUBLE_EQ(0.0, out.y_values()[1][0]);
  EXPECT_DOUBLE_EQ(-0.84170263152593661, out.y_values()[1][out.n_steps() / 2]);
  EXPECT_DOUBLE_EQ(-0.90929742675256664, out.y_values()[1][out.n_steps() - 1]);

  EXPECT_DOUBLE_EQ(-0.41614683651997603, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.90929742675256664, ystart[1]);
}

TEST_F(SimpleOscillatorTest, DenseOutputMatchesPython) {
  auto ode = Dopr5Integrator(ystart, x1, x2, atol, rtol, h1, hmin, Output{20},
                             rhs_osc);
  auto const& out = ode.stepper;

  ode.integrate();

  // Reference values generated using scipy.integrate.solve_ivp().
  auto constexpr reference_x_values = std::array{0.0,
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
  auto constexpr reference_y0_values = std::array{1.0,
                                                  0.995004165274381,
                                                  0.9800665778342761,
                                                  0.9553364891156162,
                                                  0.9210609939762899,
                                                  0.8775825618689674,
                                                  0.8253356148906368,
                                                  0.7648421872376728,
                                                  0.6967067092917164,
                                                  0.6216099682354068,
                                                  0.540302305843194,
                                                  0.4535961213685925,
                                                  0.3623577544196516,
                                                  0.2674988286034845,
                                                  0.1699671428597921,
                                                  0.07073720163741866,
                                                  -0.029199522309753125,
                                                  -0.1288444943201804,
                                                  -0.22720209468205058,
                                                  -0.32328956688476085,
                                                  -0.41614683651997625};
  auto constexpr reference_y1_values = std::array{0.0,
                                                  -0.09983341667072441,
                                                  -0.19866933081341775,
                                                  -0.29552020666230877,
                                                  -0.38941834234718053,
                                                  -0.47942553861305215,
                                                  -0.5646424733839779,
                                                  -0.6442176872563701,
                                                  -0.7173560909168458,
                                                  -0.7833269096128057,
                                                  -0.8414709847747586,
                                                  -0.8912073600416666,
                                                  -0.9320390859400902,
                                                  -0.9635581853673072,
                                                  -0.9854497299379401,
                                                  -0.9974949865471938,
                                                  -0.9995736029766767,
                                                  -0.9916648103839413,
                                                  -0.973847630807643,
                                                  -0.9463000876044179,
                                                  -0.9092974267525671};
  auto constexpr comp_tol =
      1.0e-15;  // Stricter accuracy requirement for comparison to Python
                // function that should be performing similar calculation.
  for (auto const& [ref_x, x] : vws::zip(reference_x_values, out.x_values())) {
    EXPECT_DOUBLE_EQ(ref_x, x);  // Independent variable should compare equal.
  }
  for (auto const& [ref_y0, y0] :
       vws::zip(reference_y0_values, out.y_values()[0])) {
    EXPECT_NEAR(ref_y0, y0, comp_tol);
  }
  for (auto const& [ref_y1, y1] :
       vws::zip(reference_y1_values, out.y_values()[1])) {
    EXPECT_NEAR(ref_y1, y1, comp_tol);
  }
  EXPECT_NEAR(-0.41614683651997625, ystart[0], comp_tol);
  EXPECT_NEAR(-0.9092974267525671, ystart[1], comp_tol);
}

TEST_F(SimpleOscillatorTest, DenseOutputIsConsistent) {
  auto ode = Dopr5Integrator(ystart, x1, x2, atol, rtol, h1, hmin, Output{20},
                             rhs_osc);
  auto const& out = ode.stepper;

  ode.integrate();

  // Reference values from initial run to test for consistency.
  auto constexpr reference_x_values = std::array{0.0,
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
  auto constexpr reference_y0_values = std::array{1.0,
                                                  0.995004165274381,
                                                  0.9800665778342761,
                                                  0.9553364891156162,
                                                  0.9210609939762899,
                                                  0.8775825618689674,
                                                  0.8253356148906368,
                                                  0.7648421872376728,
                                                  0.6967067092917164,
                                                  0.6216099682354068,
                                                  0.540302305843194,
                                                  0.45359612136859212,
                                                  0.36235775441965129,
                                                  0.2674988286034845,
                                                  0.1699671428597921,
                                                  0.070737201637418756,
                                                  -0.029199522309753027,
                                                  -0.12884449432018022,
                                                  -0.22720209468205022,
                                                  -0.32328956688476052,
                                                  -0.41614683651997625};
  auto constexpr reference_y1_values = std::array{0.0,
                                                  -0.09983341667072441,
                                                  -0.19866933081341775,
                                                  -0.29552020666230877,
                                                  -0.38941834234718053,
                                                  -0.47942553861305215,
                                                  -0.5646424733839779,
                                                  -0.6442176872563701,
                                                  -0.7173560909168458,
                                                  -0.7833269096128057,
                                                  -0.8414709847747586,
                                                  -0.8912073600416666,
                                                  -0.9320390859400902,
                                                  -0.9635581853673072,
                                                  -0.9854497299379401,
                                                  -0.9974949865471938,
                                                  -0.9995736029766767,
                                                  -0.99166481038394072,
                                                  -0.97384763080764247,
                                                  -0.94630008760441731,
                                                  -0.9092974267525671};
  for (auto const& [ref_x, x] : vws::zip(reference_x_values, out.x_values())) {
    EXPECT_DOUBLE_EQ(ref_x, x);
  }
  for (auto const& [ref_y0, y0] :
       vws::zip(reference_y0_values, out.y_values()[0])) {
    EXPECT_DOUBLE_EQ(ref_y0, y0);
  }
  for (auto const& [ref_y1, y1] :
       vws::zip(reference_y1_values, out.y_values()[1])) {
    EXPECT_DOUBLE_EQ(ref_y1, y1);
  }
  EXPECT_DOUBLE_EQ(-0.41614683651997625, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.9092974267525671, ystart[1]);
}

/* Need to test with non-zero starting point to ensure that bugs are not hidden
 * by default-initialization. */
TEST_F(SimpleOscillatorTest, ConsistentWithNonzeroStartingPoint) {
  auto constexpr xstart = 1.0;
  auto ode = Dopr5Integrator(ystart, xstart, x2, atol, rtol, h1, hmin,
                             Output{5}, rhs_osc);
  auto const& out = ode.stepper;

  ode.integrate();

  // Reference values from initial run to test for consistency.
  auto constexpr reference_x_values = std::array{
      1.0, 1.2, 1.3999999999999999, 1.5999999999999999, 1.7999999999999998,
      2.0};
  auto constexpr reference_y0_values = std::array{1.0,
                                                  0.9800665778342762,
                                                  0.92106099397628982,
                                                  0.82533561489063678,
                                                  0.6967067092917163,
                                                  0.54030230584326977};
  auto constexpr reference_y1_values = std::array{0.0,
                                                  -0.19866933081341756,
                                                  -0.38941834234718037,
                                                  -0.56464247338397766,
                                                  -0.71735609091684549,
                                                  -0.8414709847748435};
  for (auto const& [ref_x, x] : vws::zip(reference_x_values, out.x_values())) {
    EXPECT_DOUBLE_EQ(ref_x, x);  // independent variable should compare equal
  }
  for (auto const& [ref_y0, y0] :
       vws::zip(reference_y0_values, out.y_values()[0])) {
    EXPECT_DOUBLE_EQ(ref_y0, y0);
  }
  for (auto const& [ref_y1, y1] :
       vws::zip(reference_y1_values, out.y_values()[1])) {
    EXPECT_DOUBLE_EQ(ref_y1, y1);
  }
  EXPECT_DOUBLE_EQ(0.54030230584326955, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.84147098477484361, ystart[1]);
}