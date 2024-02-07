#pragma once

#include <vector>
#include <stdexcept>

struct Output
{
    int kmax;
    int nvar;
    int nsave;
    bool dense;
    int count;
    double x1;
    double x2;
    double xout;
    double dxout;
    std::vector<double> xsave;
    std::vector<std::vector<double>> ysave;

    Output() : kmax(-1), dense(false), count(0) {}

    Output(const int nsavee) : kmax(500), nsave(nsavee), count(0), xsave(kmax)
    {
        dense = nsave > 0 ? true : false;
    }

    void init(const int neqn, const double xlo, const double xhi)
    {
        nvar = neqn;
        if (kmax == -1) { return; }
        ysave.resize(nvar, kmax);
        if (dense)
        {
            x1 = xlo;
            x2 = xhi;
            xout = x1;
            dxout = (x2 - x1) / nsave;
        }
    }

    void resize()
    {
        int kold = kmax;
        kmax *= 2;
        std::vector<double> tempvec(xsave);
        xsave.resize(kmax);
        for (int k = 0; k < kold; ++k)
        {
            xsave[k] = tempvec[k];
        }
        std::vector<std::vector<double>> tempmat(ysave);
        ysave.resize(nvar);
        for (int i = 0; i < nvar; ++i)
        {
            ysave[i].resize(kmax);
            for (int k = 0; k < kold; ++k)
            {
                ysave[i][k] = tempmat[i][k];
            }
        }
    }

    template <typename Stepper>
    void save_dense(Stepper& stepper, const double xout, const double h)
    {
        if (count == kmax)
        {
            resize();
        }
        for (int i = 0; i < nvar; ++i)
        {
            ysave[i][count] = stepper.dense_out(i, xout, h);
        }
        xsave[count++] = xout;
    }

    void save(const double x, std::vector<double>& y)
    {
        if (kmax <= 0) { return; }
        if (count == kmax)
        {
            resize();
        }
        for (int i = 0; i < nvar; ++i)
        {
            ysave[i][count] = y[i];
        }
        xsave[count++] = x;
    }

    template <typename Stepper>
    void out(const int nstp, const double x, std::vector<double>& y,
        Stepper& stepper, const double h)
    {
        if (!dense)
        {
            throw std::runtime_error("dense output not set in Output");
        }
        if (nstp == -1)
        {
            save(x, y);
            xout += dxout;
        }
        else
        {
            while ((x - xout) * (x2 - x1) > 0.0)
            {
                save_dense(stepper, xout, h);
                xout += dxout;
            }
        }
    }
};