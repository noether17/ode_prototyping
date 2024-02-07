#include "ODEIntegrator.hpp"
#include "StepperDopr5.hpp"

#include <gtest/gtest.h>

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

    // Values take from initial run. Testing for consistency.
    EXPECT_DOUBLE_EQ( 1.7644320190605265,  ystart[0]);
    EXPECT_DOUBLE_EQ(-0.83427005245677999, ystart[1]);
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