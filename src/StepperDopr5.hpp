#pragma once

#include <cmath>
#include <vector>

#include "StepperBase.hpp"

template <typename D>
struct StepperDopr5 : StepperBase
{
    typedef D Dtype;
    std::vector<double> k2;
    std::vector<double> k3;
    std::vector<double> k4;
    std::vector<double> k5;
    std::vector<double> k6;
    std::vector<double> rcont1;
    std::vector<double> rcont2;
    std::vector<double> rcont3;
    std::vector<double> rcont4;
    std::vector<double> rcont5;
    std::vector<double> dydxnew;
    
    StepperDopr5(std::vector<double>& yy, std::vector<double>& dydxx, double& xx,
        const double atoll, const double rtoll, bool dens);
    
    void step(const double htry, D& derivs);
    void dy(const double h, D& derivs);
    void prepare_dense(const double h, D& derivs);
    double dense_out(const int i, const double x, const double h);
    double error();
    struct Controller
    {
        double hnext;
        double errold;
        bool reject;
        
        Controller();
        bool success(const double err, double& h);
    };
    Controller con;
};

template <typename D>
StepperDopr5<D>::StepperDopr5(std::vector<double>& yy, std::vector<double>& dydxx,
    double& xx, const double atoll, const double rtoll, bool dens)
    : StepperBase(yy, dydxx, xx, atoll, rtoll, dens), k2(n), k3(n), k4(n), k5(n),
      k6(n), rcont1(n), rcont2(n), rcont3(n), rcont4(n), rcont5(n), dydxnew(n)
{
    eps = std::numeric_limits<double>::epsilon();
}

template <typename D>
void StepperDopr5<D>::step(const double htry, D& derivs)
{
    double h = htry;
    for (;;)
    {
        dy(h, derivs);
        const double err = error();
        if (con.success(err, h))
        {
            break;
        }
        if (fabs(h) <= fabs(x)*eps)
        {
            throw std::runtime_error("step size underflow in StepperDopr5");
        }
    }
    if (dense)
    {
        prepare_dense(h, derivs);
    }
    dydx = dydxnew;
    y = yout;
    xold = x;
    x += (hdid = h);
    hnext = con.hnext;
}

template <typename D>
void StepperDopr5<D>::dy(const double h, D& derivs)
{
    static const double c2 = 0.2;
    static const double c3 = 0.3;
    static const double c4 = 0.8;
    static const double c5 = 8.0/9.0;
    static const double a21 = 0.2;
    static const double a31 = 3.0/40.0;
    static const double a32 = 9.0/40.0;
    static const double a41 = 44.0/45.0;
    static const double a42 = -56.0/15.0;
    static const double a43 = 32.0/9.0;
    static const double a51 = 19372.0/6561.0;
    static const double a52 = -25360.0/2187.0;
    static const double a53 = 64448.0/6561.0;
    static const double a54 = -212.0/729.0;
    static const double a61 = 9017.0/3168.0;
    static const double a62 = -355.0/33.0;
    static const double a63 = 46732.0/5247.0;
    static const double a64 = 49.0/176.0;
    static const double a65 = -5103.0/18656.0;
    static const double a71 = 35.0/384.0;
    static const double a73 = 500.0/1113.0;
    static const double a74 = 125.0/192.0;
    static const double a75 = -2187.0/6784.0;
    static const double a76 = 11.0/84.0;
    static const double e1 = 71.0/57600.0;
    static const double e3 = -71.0/16695.0;
    static const double e4 = 71.0/1920.0;
    static const double e5 = -17253.0/339200.0;
    static const double e6 = 22.0 / 525.0;
    static const double e7 = -1.0/40.0;

    std::vector<double> ytemp(n);
    int i;
    for (i = 0; i < n; ++i)
    {
        ytemp[i] = y[i] + h*a21*dydx[i];
    }
    derivs(x + c2*h, ytemp, k2);
    for (i = 0; i < n; ++i)
    {
        ytemp[i] = y[i] + h*(a31*dydx[i] + a32*k2[i]);
    }
    derivs(x + c3*h, ytemp, k3);
    for (i = 0; i < n; ++i)
    {
        ytemp[i] = y[i] + h*(a41*dydx[i] + a42*k2[i] + a43*k3[i]);
    }
    derivs(x + c4*h, ytemp, k4);
    for (i = 0; i < n; ++i)
    {
        ytemp[i] = y[i] + h*(a51*dydx[i] + a52*k2[i] + a53*k3[i] + a54*k4[i]);
    }
    derivs(x + c5*h, ytemp, k5);
    for (i = 0; i < n; ++i)
    {
        ytemp[i] = y[i] + h*(a61*dydx[i] + a62*k2[i] + a63*k3[i] + a64*k4[i] + a65*k5[i]);
    }
    double xph = x + h;
    derivs(xph, ytemp, k6);
    for (i = 0; i < n; ++i)
    {
        yout[i] = y[i] + h*(a71*dydx[i] + a73*k3[i] + a74*k4[i] + a75*k5[i] + a76*k6[i]);
    }
    derivs(xph, yout, dydxnew);
    for (i = 0; i < n; ++i)
    {
        yerr[i] = h*(e1*dydx[i] + e3*k3[i] + e4*k4[i] + e5*k5[i] + e6*k6[i] + e7*dydxnew[i]);
    }
}

template <typename D>
void StepperDopr5<D>::prepare_dense(const double h, D& derivs)
{
    std::vector<double> ytemp(n);
    static const double d1 = -12715105075.0/11282082432.0;
    static const double d3 = 87487479700.0/32700410799.0;
    static const double d4 = -10690763975.0/1880347072.0;
    static const double d5 = 701980252875.0/199316789632.0;
    static const double d6 = -1453857185.0 / 822651844.0;
    static const double d7 = 69997945.0 / 29380423.0;
    for (int i = 0; i < n; ++i)
    {
        rcont1[i] = y[i];
        double ydiff = yout[i] - y[i];
        rcont2[i] = ydiff;
        double bspl = h*dydx[i] - ydiff;
        rcont3[i] = bspl;
        rcont4[i] = ydiff - h*dxdynew[i] - bspl;
        rcont5[i] = h*(d1*dydx[i] + d3*k3[i] + d4*k4[i] + d5*k5[i] + d6*k6[i] + d7*dydxnew[i]);
    }
}

template <typename D>
double StepperDopr5<D>::dense_out(const int i, const double x, const double h)
{
    double s = (x - xold)/h;
    double s1 = 1.0 - s;
    return rcont1[i] + s*(rcont2[i] + s1*(rcont3[i] + s*(rcont4[i] + s1*rcont5[i])));
}

template <typename D>
double StepperDopr5<D>::error()
{
    double err = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double sk = atol + rtol*std::max(fabs(y[i]), fabs(yout[i]));
        err += pow(yerr[i]/sk, 2);
    }
    return sqrt(err/n);
}

template <typename D>
StepperDopr5<D>::Controller::Controller()
    : reject(false), errold(1.0e-4)
{}

template <typename D>
StepperDopr5<D>::Controller::success(const double err, double& h)
{
    static const double beta = 0.0;
    static const double alpha = 0.2 - beta*0.75;
    static const double safe = 0.9;
    static const double minscale = 0.2;
    static const double maxscale = 10.0;
    double scale;
    if (err <= 1.0)
    {
        if (err == 0.0)
        {
            scale = maxscale;
        }
        else
        {
            scale = safe*pow(err, -alpha)*pow(errold, beta);
            if (scale < minscale) { scale = minscale; }
            if (scale > maxscale) { scale = maxscale; }
        }
        if (reject)
        {
            hnext = h*min(scale, 1.0);
        }
        else
        {
            hnext = h*scale;
        }
        errold = max(err, 1.0e-4);
        reject = false;
        return true;
    }
    else
    {
        scale = max(safe*pow(err, -alpha), minscale);
        h *= scale;
        reject = true;
        return false;
    }
}