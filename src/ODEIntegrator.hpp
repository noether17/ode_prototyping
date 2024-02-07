#pragma once

#include <vector>

#include "Output.hpp"

template <typename Stepper>
struct ODEIntegrator
{
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

    ODEIntegrator(std::vector<double>& ystartt, const double xx1, const double xx2,
        const double atol, const double rtol, const double h1, const double hminn,
        Output& outt, typename Stepper::Dtype& derivss);
    
    void integrate();
};