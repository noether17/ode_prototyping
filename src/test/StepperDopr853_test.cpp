#include "StepperDopr853.hpp"

#include <gtest/gtest.h>

#include <array>
#include <ranges>
#include <vector>

#include "ODEIntegrator.hpp"

namespace vws = std::views;

class VanDerPolTest : public testing::Test {
 protected:
  static auto constexpr nvar = 2;
  static auto constexpr atol = 1.0e-3;
  static auto constexpr rtol = atol;
  static auto constexpr h1 = 0.01;
  static auto constexpr hmin = 0.0;
  static auto constexpr x1 = 0.0;
  static auto constexpr x2 = 2.0;
  static auto constexpr eps = 1.0e-3;
  static auto constexpr rhs_van = [](double, std::vector<double> const& y,
                                     std::vector<double>& dydx) {
    dydx[0] = y[1];
    dydx[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps;
  };

  std::vector<double> ystart{2.0, 0.0};

  using Dopr853IntegratorNoOutput =
      ODEIntegrator<StepperDopr853<decltype(rhs_van), NoOutput>>;
  using Dopr853IntegratorRawOutput =
      ODEIntegrator<StepperDopr853<decltype(rhs_van), RawOutput>>;
  using Dopr853IntegratorDenseOutput =
      ODEIntegrator<StepperDopr853Dense<decltype(rhs_van), DenseOutput>>;
};

TEST_F(VanDerPolTest, StepperDopr853DefaultOutputCtorSuppressesOutput) {
  auto ode =
      Dopr853IntegratorNoOutput(ystart, x1, x2, atol, rtol, h1, hmin, rhs_van);

  ode.integrate();

  EXPECT_DOUBLE_EQ(1.7636585542718883, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.8354674509696991, ystart[1]);
}

/* Consistency tests (testing for double equality) are to ensure no accidental
 * algorithm changes are made during refactoring. These tests are far stricter
 * than the actual requirements. If an intentional change in algorithm results
 * in small differences in output, these values may be updated. */
TEST_F(VanDerPolTest, StepperDopr853ActualIntegrationStepsAreConsistent) {
  auto ode =
      Dopr853IntegratorRawOutput(ystart, x1, x2, atol, rtol, h1, hmin, rhs_van);
  auto const& out = ode.stepper;

  ode.integrate();

  EXPECT_EQ(614, out.n_steps());

  EXPECT_DOUBLE_EQ(0.0, out.x_values()[0]);
  EXPECT_DOUBLE_EQ(0.98638788529248478, out.x_values()[out.n_steps() / 2]);
  EXPECT_DOUBLE_EQ(2.0, out.x_values()[out.n_steps() - 1]);

  EXPECT_DOUBLE_EQ(2.0, out.y_values()[0][0]);
  EXPECT_DOUBLE_EQ(-1.8983524659069262, out.y_values()[0][out.n_steps() / 2]);
  EXPECT_DOUBLE_EQ(1.7636585542718883, out.y_values()[0][out.n_steps() - 1]);

  EXPECT_DOUBLE_EQ(0.0, out.y_values()[1][0]);
  EXPECT_DOUBLE_EQ(0.72695745382982302, out.y_values()[1][out.n_steps() / 2]);
  EXPECT_DOUBLE_EQ(-0.8354674509696991, out.y_values()[1][out.n_steps() - 1]);

  EXPECT_DOUBLE_EQ(1.7636585542718883, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.8354674509696991, ystart[1]);
}

TEST_F(VanDerPolTest, StepperDopr853DenseOutputMatchesPython) {
  auto ode = Dopr853IntegratorDenseOutput(ystart, x1, x2, atol, rtol, h1, hmin,
                                          rhs_van);
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
  auto constexpr reference_y0_values = std::array{2.0,
                                                  1.9316122783182301,
                                                  1.858478721489275,
                                                  1.7797321406396824,
                                                  1.6936478062587523,
                                                  1.5973048438123825,
                                                  1.48539617970096,
                                                  1.344508188518583,
                                                  1.1034118596992013,
                                                  -1.9595852384343717,
                                                  -1.8883998282486796,
                                                  -1.8120698785117855,
                                                  -1.7291573370452291,
                                                  -1.6373650883017334,
                                                  -1.5326873072225298,
                                                  -1.406032003621479,
                                                  -1.2275084897783375,
                                                  1.9872064158183738,
                                                  1.9177885756335038,
                                                  1.843680289685086,
                                                  1.7636574228798523};
  auto constexpr reference_y1_values = std::array{0.0,
                                                  -0.7094442261327178,
                                                  -0.7202254893938206,
                                                  -0.8068650706654498,
                                                  -0.9389484552331121,
                                                  -0.9994329225021683,
                                                  -1.224640028673981,
                                                  -1.6518197427464374,
                                                  -4.619317342577298,
                                                  0.6918163442868651,
                                                  0.7734155795544163,
                                                  0.8304960343582835,
                                                  0.8815211399286773,
                                                  0.9417531200490238,
                                                  1.1765597740458502,
                                                  1.3938534632445547,
                                                  2.321437090654019,
                                                  -0.6689116264185984,
                                                  -0.7061440632836788,
                                                  -0.7646734056087162,
                                                  -0.8354687358368221};
  auto constexpr comp_tol =
      1.0e-4;  // Stricter accuracy requirement for comparison to Python
               // function that should be performing similar calculation.
  for (const auto& [ref_x, x] : vws::zip(reference_x_values, out.x_values())) {
    EXPECT_DOUBLE_EQ(ref_x, x);  // Independent variable should compare equal.
  }
  for (const auto& [ref_y0, y0] :
       vws::zip(reference_y0_values, out.y_values()[0])) {
    EXPECT_NEAR(ref_y0, y0, comp_tol);
  }
  for (const auto& [ref_y1, y1] :
       vws::zip(reference_y1_values, out.y_values()[1])) {
    EXPECT_NEAR(ref_y1, y1, comp_tol);
  }
  EXPECT_NEAR(1.7636574228798523, ystart[0], comp_tol);
  EXPECT_NEAR(-0.8354687358368221, ystart[1], comp_tol);
}

TEST_F(VanDerPolTest, StepperDopr853DenseOutputIsConsistent) {
  auto ode = Dopr853IntegratorDenseOutput(ystart, x1, x2, atol, rtol, h1, hmin,
                                          rhs_van);
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
  auto constexpr reference_y0_values = std::array{2.0,
                                                  1.9316122783182343,
                                                  1.8584787214893352,
                                                  1.7797321406395954,
                                                  1.6936478062588005,
                                                  1.5973048438124426,
                                                  1.4853961797009534,
                                                  1.3445081885185826,
                                                  1.1034118596940761,
                                                  -1.9595852385244072,
                                                  -1.8883998283360151,
                                                  -1.8120698785806237,
                                                  -1.7291573373244966,
                                                  -1.6373650886237239,
                                                  -1.53268730742399,
                                                  -1.4060320034274552,
                                                  -1.2275084884890473,
                                                  1.9872073270417665,
                                                  1.9177895398084071,
                                                  1.8436813288440266,
                                                  1.7636585542718883};
  auto constexpr reference_y1_values = std::array{0.0,
                                                  -0.7094442261444629,
                                                  -0.72022548954393439,
                                                  -0.80686507047834188,
                                                  -0.93894845532670956,
                                                  -0.99943292259922056,
                                                  -1.2246400286700401,
                                                  -1.6518197427494905,
                                                  -4.6193173417203504,
                                                  0.69181635473357528,
                                                  0.77341556725241589,
                                                  0.83049596472069132,
                                                  0.88152147969282724,
                                                  0.94175345678023936,
                                                  1.17655985351827,
                                                  1.393853095827446,
                                                  2.3214362560484116,
                                                  -0.66890691862254825,
                                                  -0.70612818420425061,
                                                  -0.76466904126707647,
                                                  -0.8354674509696991};
  for (const auto& [ref_x, x] : vws::zip(reference_x_values, out.x_values())) {
    EXPECT_DOUBLE_EQ(ref_x, x);
  }
  for (const auto& [ref_y0, y0] :
       vws::zip(reference_y0_values, out.y_values()[0])) {
    EXPECT_DOUBLE_EQ(ref_y0, y0);
  }
  for (const auto& [ref_y1, y1] :
       vws::zip(reference_y1_values, out.y_values()[1])) {
    EXPECT_DOUBLE_EQ(ref_y1, y1);
  }
  EXPECT_DOUBLE_EQ(1.7636585542718883, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.8354674509696991, ystart[1]);
}

class SimpleOscillatorTest : public testing::Test {
 protected:
  static auto constexpr nvar = 2;
  static auto constexpr atol = 1.0e-10;
  static auto constexpr rtol = atol;
  static auto constexpr h1 = 0.01;
  static auto constexpr hmin = 0.0;
  static auto constexpr x1 = 0.0;
  static auto constexpr x2 = 2.0;
  static auto constexpr rhs_osc = [](double, std::vector<double>& y,
                                     std::vector<double>& dydx) {
    dydx[0] = y[1];
    dydx[1] = -y[0];
  };

  std::vector<double> ystart{1.0, 0.0};

  using Dopr853IntegratorRawOutput =
      ODEIntegrator<StepperDopr853<decltype(rhs_osc), RawOutput>>;
  using Dopr853IntegratorDenseOutput =
      ODEIntegrator<StepperDopr853Dense<decltype(rhs_osc), DenseOutput>>;
};

/* Consistency tests (testing for double equality) are to ensure no accidental
 * algorithm changes are made during refactoring. These tests are far stricter
 * than the actual requirements. If an intentional change in algorithm results
 * in small differences in output, these values may be updated. */
TEST_F(SimpleOscillatorTest,
       StepperDopr853ActualIntegrationStepsAreConsistent) {
  auto ode =
      Dopr853IntegratorRawOutput(ystart, x1, x2, atol, rtol, h1, hmin, rhs_osc);
  auto const& out = ode.stepper;

  ode.integrate();

  EXPECT_EQ(9, out.n_steps());

  EXPECT_DOUBLE_EQ(0.0, out.x_values()[0]);
  EXPECT_DOUBLE_EQ(0.79032602061558577, out.x_values()[out.n_steps() / 2]);
  EXPECT_DOUBLE_EQ(2.0, out.x_values()[out.n_steps() - 1]);

  EXPECT_DOUBLE_EQ(1.0, out.y_values()[0][0]);
  EXPECT_DOUBLE_EQ(0.7036136884476063, out.y_values()[0][out.n_steps() / 2]);
  EXPECT_DOUBLE_EQ(-0.41614683651801287, out.y_values()[0][out.n_steps() - 1]);

  EXPECT_DOUBLE_EQ(0.0, out.y_values()[1][0]);
  EXPECT_DOUBLE_EQ(-0.71058270273473667, out.y_values()[1][out.n_steps() / 2]);
  EXPECT_DOUBLE_EQ(-0.90929742683477, out.y_values()[1][out.n_steps() - 1]);

  EXPECT_DOUBLE_EQ(-0.41614683651801287, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.90929742683477, ystart[1]);
}

TEST_F(SimpleOscillatorTest, StepperDopr853DenseOutputMatchesPython) {
  auto ode = Dopr853IntegratorDenseOutput(ystart, x1, x2, atol, rtol, h1, hmin,
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
                                                  0.995004165278017,
                                                  0.9800665778463475,
                                                  0.9553364893357608,
                                                  0.9210609936665004,
                                                  0.877582561901116,
                                                  0.8253356149983048,
                                                  0.764842187327496,
                                                  0.6967067091812618,
                                                  0.6216099682773553,
                                                  0.540302306001837,
                                                  0.45359612128624477,
                                                  0.36235775449467666,
                                                  0.26749882865648245,
                                                  0.16996714295479937,
                                                  0.07073720161658875,
                                                  -0.029199522277303797,
                                                  -0.12884449427687886,
                                                  -0.22720209465511865,
                                                  -0.32328956683367616,
                                                  -0.4161468365180746};
  auto constexpr reference_y1_values = std::array{0.0,
                                                  -0.0998334166468278,
                                                  -0.1986693307965235,
                                                  -0.29552020669799695,
                                                  -0.38941834223855837,
                                                  -0.4794255386037429,
                                                  -0.5646424734404543,
                                                  -0.6442176872509984,
                                                  -0.717356090782937,
                                                  -0.7833269096160714,
                                                  -0.8414709849538273,
                                                  -0.8912073598534077,
                                                  -0.9320390859620191,
                                                  -0.9635581854537687,
                                                  -0.9854497300854524,
                                                  -0.9974949863614198,
                                                  -0.9995736030386739,
                                                  -0.9916648106240408,
                                                  -0.9738476306960134,
                                                  -0.9463000876935369,
                                                  -0.9092974268347512};
  auto constexpr comp_tol =
      1.0e-9;  // Stricter accuracy requirement for comparison to Python
               // function that should be performing similar calculation.
  for (const auto& [ref_x, x] : vws::zip(reference_x_values, out.x_values())) {
    EXPECT_DOUBLE_EQ(ref_x, x);  // Independent variable should compare equal.
  }
  for (const auto& [ref_y0, y0] :
       vws::zip(reference_y0_values, out.y_values()[0])) {
    EXPECT_NEAR(ref_y0, y0, comp_tol);
  }
  for (const auto& [ref_y1, y1] :
       vws::zip(reference_y1_values, out.y_values()[1])) {
    EXPECT_NEAR(ref_y1, y1, comp_tol);
  }
  EXPECT_NEAR(-0.41614683651997625, ystart[0], comp_tol);
  EXPECT_NEAR(-0.9092974267525671, ystart[1], comp_tol);
}

TEST_F(SimpleOscillatorTest, StepperDopr853DenseOutputIsConsistent) {
  auto ode = Dopr853IntegratorDenseOutput(ystart, x1, x2, atol, rtol, h1, hmin,
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
                                                  0.99500416528781477,
                                                  0.98006657793938101,
                                                  0.95533648916707048,
                                                  0.92106099380834583,
                                                  0.87758256188944672,
                                                  0.82533561509979825,
                                                  0.7648421870658686,
                                                  0.69670670935715528,
                                                  0.62160996830538506,
                                                  0.54030230597309425,
                                                  0.45359612124685283,
                                                  0.36235775449376306,
                                                  0.26749882869694513,
                                                  0.16996714287483936,
                                                  0.070737201691768836,
                                                  -0.029199522276831924,
                                                  -0.12884449426873257,
                                                  -0.22720209466067331,
                                                  -0.32328956683362298,
                                                  -0.41614683651801287};
  auto constexpr reference_y1_values = std::array{0.0,
                                                  -0.099833416648732465,
                                                  -0.19866933080879839,
                                                  -0.29552020666300693,
                                                  -0.38941834227306915,
                                                  -0.47942553859720011,
                                                  -0.56464247349047925,
                                                  -0.64421768710214034,
                                                  -0.71735609089146357,
                                                  -0.78332690964802465,
                                                  -0.84147098490570094,
                                                  -0.89120735982110444,
                                                  -0.93203908596047802,
                                                  -0.96355818555607042,
                                                  -0.98544972984832069,
                                                  -0.99749498659822255,
                                                  -0.99957360305615339,
                                                  -0.9916648106167375,
                                                  -0.97384763057852497,
                                                  -0.94630008769357532,
                                                  -0.90929742683477};
  for (const auto& [ref_x, x] : vws::zip(reference_x_values, out.x_values())) {
    EXPECT_DOUBLE_EQ(ref_x, x);
  }
  for (const auto& [ref_y0, y0] :
       vws::zip(reference_y0_values, out.y_values()[0])) {
    EXPECT_DOUBLE_EQ(ref_y0, y0);
  }
  for (const auto& [ref_y1, y1] :
       vws::zip(reference_y1_values, out.y_values()[1])) {
    EXPECT_DOUBLE_EQ(ref_y1, y1);
  }
  EXPECT_DOUBLE_EQ(-0.41614683651801287, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.90929742683477, ystart[1]);
}

/* Need to test with non-zero starting point to ensure that bugs are not hidden
 * by default-initialization. */
TEST_F(SimpleOscillatorTest, ConsistentWithNonzeroStartingPoint) {
  auto constexpr xstart = 1.0;
  auto ode = Dopr853IntegratorDenseOutput(ystart, xstart, x2, atol, rtol, h1,
                                          hmin, rhs_osc);
  ode.stepper.set_n_intervals(5);
  auto const& out = ode.stepper;

  ode.integrate();

  // Reference values from initial run to test for consistency.
  auto constexpr reference_x_values = std::array{
      1.0, 1.2, 1.3999999999999999, 1.5999999999999999, 1.7999999999999998,
      2.0};
  auto constexpr reference_y0_values = std::array{1.0,
                                                  0.98006657793938112,
                                                  0.92106099380834583,
                                                  0.82533561509979836,
                                                  0.69670670935540835,
                                                  0.54030230587824779};
  auto constexpr reference_y1_values = std::array{0.0,
                                                  -0.19866933080879828,
                                                  -0.38941834227306893,
                                                  -0.56464247349047914,
                                                  -0.71735609088929897,
                                                  -0.84147098479941029};
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
  EXPECT_DOUBLE_EQ(0.54030230587824757, ystart[0]);
  EXPECT_DOUBLE_EQ(-0.8414709847994104, ystart[1]);
}