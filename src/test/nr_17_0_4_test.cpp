#include "ODEIntegrator.hpp"
#include "StepperDopr5.hpp"

#include <ranges>

#include <gtest/gtest.h>

namespace vws = std::views;

struct RHSVan
{
    double eps;
    RHSVan(double epss) : eps(epss) {}
    void operator()(const double x, std::vector<double>& y, std::vector<double>& dydx)
    {
        dydx[0] = y[1];
        dydx[1] = ((1.0 - y[0]*y[0])*y[1] - y[0]) / eps;
    }
};

TEST(ODEIntegratorTest, VanDerPolTest)
{
    const int nvar = 2;
    const double atol = 1.0e-3;
    const double rtol = atol;
    const double h1 = 0.01;
    const double hmin = 0.0;
    const double x1 = 0.0;
    const double x2 = 2.0;
    std::vector<double> ystart(nvar);
    ystart[0] = 2.0;
    ystart[1] = 0.0;
    Output out(20);
    RHSVan d(1.0e-3);
    ODEIntegrator<StepperDopr5<RHSVan>> ode(ystart, x1, x2, atol, rtol, h1,
        hmin, out, d);

    ode.integrate();

    // Reference values generated using scipy.integrate.solve_ivp
    const auto reference_x_values = std::vector{ 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6,
        0.7, 0.7999999999999999, 0.8999999999999999, 0.9999999999999999, 1.0999999999999999, 1.2,
        1.3, 1.4000000000000001, 1.5000000000000002, 1.6000000000000003, 1.7000000000000004,
        1.8000000000000005, 1.9000000000000006, 2.0 };
    const auto reference_y0_values = std::vector{ 2.0, 1.9316114668840196, 1.8584936578161697,
        1.779738838649533, 1.6936304673029232, 1.5973233455330436, 1.4853988650659193,
        1.3445145377549454, 1.1027432107238924, -1.9607574926871358, -1.889636370708954,
        -1.813401592162303, -1.730627659606652, -1.639036082901385, -1.534583204376312,
        -1.4085081361939422, -1.2316250330343574, 1.9878333637317933, 1.9184567233175167,
        1.8443946860836433, 1.7644320186384204 };
    const auto reference_y1_values = std::vector{ 0.0, -0.70723585006735, -0.7569013815134848,
        -0.8214089752926855, -0.9065846314294861, -1.0280880445964888, -1.2278827867432427,
        -1.6569111260368745, -4.48537067895128, 0.6896276917781045, 0.7363385787399728,
        0.79157138516551, 0.868805850565066, 0.9688135932292358, 1.131427622338704,
        1.4281071047886105, 2.339031986623704, -0.6741234554050665, -0.7162074462396393,
        -0.7682889897929661, -0.8342700528712028 };
    constexpr auto dep_var_tol = 1.0e-9; // stricter accuracy requirement for comparison to Python
                                         // function that should be performing similar calculation
    for (const auto& [ref_x, x] : vws::zip(reference_x_values, out.xsave))
    {
        EXPECT_DOUBLE_EQ(ref_x, x); // independent variable should compare equal
    }
    for (const auto& [ref_y0, y0] : vws::zip(reference_y0_values, out.ysave[0]))
    {
        EXPECT_NEAR(ref_y0, y0, dep_var_tol);
    }
    for (const auto& [ref_y1, y1] : vws::zip(reference_y1_values, out.ysave[1]))
    {
        EXPECT_NEAR(ref_y1, y1, dep_var_tol);
    }
    EXPECT_NEAR(1.7644320186384204, ystart[0], dep_var_tol);
    EXPECT_NEAR(-0.8342700528712028, ystart[1], dep_var_tol);
}

struct RHSOsc
{
    void operator()(const double x, std::vector<double>& y, std::vector<double>& dydx)
    {
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
};

TEST(ODEIntegratorTest, SimpleOscillatorTest)
{
    const int nvar = 2;
    const double atol = 1.0e-10;
    const double rtol = atol;
    const double h1 = 0.01;
    const double hmin = 0.0;
    const double x1 = 0.0;
    const double x2 = 2.0;
    std::vector<double> ystart(nvar);
    ystart[0] = 1.0;
    ystart[1] = 0.0;
    Output out(20);
    RHSOsc d;
    ODEIntegrator<StepperDopr5<RHSOsc>> ode(ystart, x1, x2, atol, rtol, h1,
        hmin, out, d);

    ode.integrate();

    // Values take from initial run. Testing for consistency.
    EXPECT_NEAR(-0.4161468365, ystart[0], atol);
    EXPECT_NEAR(-0.9092974268, ystart[1], atol);
}