import numpy as np
import scipy.integrate as spi

nvar = 2
atol = 1.0e-3
rtol = 1.0e-3
h1 = 0.01
hmin = 0.0
x1 = 0.0
x2 = 2.0
dense = True
nsave = 20
dxout = (x2 - x1) / nsave
x_eval = [x1]
for i in np.arange(nsave):
    x_eval.append(x_eval[i] + dxout)
x_eval[-1] = x2

def main():
    sol = spi.solve_ivp(RHSVan, [x1, x2], [2, 0], method='RK45', atol=atol, rtol=rtol, first_step=h1,
                        dense_output=dense, t_eval=x_eval)
    np.set_printoptions(floatmode='unique')
    print(sol.t)
    print(sol.y)

def RHSVan(t, y):
    eps = 1.0e-3
    return np.array([y[1], ((1.0 - y[0]**2)*y[1] - y[0]) / eps])

if __name__ == "__main__":
    main()