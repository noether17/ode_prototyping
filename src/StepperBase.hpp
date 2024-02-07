#pragma once

#include <vector>

struct StepperBase
{
    double& x;
    double xold;
    std::vector<double>& y;
    std::vector<double>& dydx;
    double atol;
    double rtol;
    bool dense;
    double hdid;
    double hnext;
    double eps;
    int n;
    int neqn;
    std::vector<double> yout;
    std::vector<double> yerr;

    StepperBase(std::vector<double>& yy, std::vector<double>& dydxx, double& xx,
        const double atoll, const double rtoll, bool dens)
        : x(xx), y(yy), dydx(dydxx), atol(atoll), rtol(rtoll), dense(dens),
          n(y.size()), neqn(n), yout(n), yerr(n)
    {}
};