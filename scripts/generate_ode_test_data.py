import numpy as np
import scipy.integrate as spi

nvar = 2
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
    sol = spi.solve_ivp(RHSVan, [x1, x2], [2.0, 0.0], method='RK45', atol=1.0e-3, rtol=1.0e-3, first_step=h1,
                        dense_output=dense, t_eval=x_eval)
    np.set_printoptions(floatmode='unique')
    print("Van der Pol")
    print_initializer_list(sol.t)
    print_initializer_list(sol.y[0])
    print_initializer_list(sol.y[1])

    sol = spi.solve_ivp(RHSOsc, [x1, x2], [1.0, 0.0], method='RK45', atol=1.0e-10, rtol=1.0e-10, first_step=h1,
                        dense_output=dense, t_eval=x_eval)
    np.set_printoptions(floatmode='unique')
    print("Oscillator")
    print_initializer_list(sol.t)
    print_initializer_list(sol.y[0])
    print_initializer_list(sol.y[1])

def RHSVan(t, y):
    eps = 1.0e-3
    return np.array([y[1], ((1.0 - y[0]**2)*y[1] - y[0]) / eps])

def RHSOsc(t, y):
    return np.array([y[1], -y[0]])

def print_initializer_list(array):
    print('{ ', end='')
    for i in np.arange(len(array)):
        print(array[i], end='')
        if i < len(array) - 1:
            print(', ', end='')
    print(' }')

if __name__ == "__main__":
    main()