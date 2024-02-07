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
}