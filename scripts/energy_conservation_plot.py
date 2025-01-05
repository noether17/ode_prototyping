import matplotlib.pyplot as plt
import numpy as np

def main():
    for softening in data.keys():
        dE_E = np.array([np.mean(values) for values in data[softening].values()])
        plt.loglog(data[softening].keys(), dE_E, label=f"{softening}")
    plt.xlabel("Tolerance Parameter")
    plt.ylabel(r"Mean $\frac{E(t) - E_0}{E_0}$")
    plt.legend()
    plt.show()

# data collected with plot_conserved_quantities.py for SpinningParticlesInBox scenario.
# {softening: {tolerance: [values of dE/E0 for three simulation runs]}}
data = {2.48e-4: {1.0e-3: np.array([4.42e-1, 1.30e-1, 5.74e-3]),
                  3.0e-4: np.array([2.83e+1, 1.07e+0, 1.29e-1]),
                  1.0e-4: np.array([8.03e+1, 4.20e+0, 4.22e+0]),
                  3.0e-5: np.array([7.25e-1, 1.66e-4, 1.35e-3]),
                  1.0e-5: np.array([1.71e+1, 2.06e-4, 1.98e-4]),
                  3.0e-6: np.array([1.78e-4, 1.87e-4, 1.84e-4]),
                  1.0e-6: np.array([2.05e-4, 2.12e-4, 1.92e-4]),
                  3.0e-7: np.array([2.28e-4, 1.73e-4, 2.31e-4]),
                  1.0e-7: np.array([2.00e-4, 1.93e-4, 2.06e-4])},
        8.27e-5: {1.0e-3: np.array([9.09e-2, 2.38e-1, 1.23e-2]),
                  3.0e-4: np.array([3.86e-1, 1.18e+0, 2.63e-3]),
                  1.0e-4: np.array([3.43e-3, 1.12e-2, 1.24e+0]),
                  3.0e-5: np.array([5.12e-5, 5.47e-5, 3.41e+1]),
                  1.0e-5: np.array([6.50e-5, 5.48e-5, 5.51e-5]),
                  3.0e-6: np.array([5.93e-5, 6.44e-5, 5.84e-5]),
                  1.0e-6: np.array([6.65e-5, 6.39e-5, 6.71e-5]),
                  3.0e-7: np.array([6.48e-5, 6.87e-5, 6.14e-5]),
                  1.0e-7: np.array([6.75e-5, 7.84e-5, 7.46e-5])},
        2.48e-5: {1.0e-3: np.array([1.72e-4, 3.13e-3, 6.01e-3]),
                  3.0e-4: np.array([1.99e-1, 3.00e-3, 2.56e-4]),
                  1.0e-4: np.array([2.39e-5, 1.60e-4, 9.96e-1]),
                  3.0e-5: np.array([4.13e+0, 3.48e-2, 1.96e-5]),
                  1.0e-5: np.array([2.11e-5, 1.78e-5, 1.94e-5]),
                  3.0e-6: np.array([1.90e-5, 1.22e-5, 2.92e-5]),
                  1.0e-6: np.array([1.93e-5, 2.17e-5, 1.83e-5]),
                  3.0e-7: np.array([2.02e-5, 2.13e-5, 1.85e-5]),
                  1.0e-7: np.array([2.40e-5, 2.17e-5, 2.36e-5])},
        2.48e-6: {1.0e-3: np.array([1.64e-4, 8.23e-5, 4.72e-3]),
                  3.0e-4: np.array([1.24e-4, 5.88e+0, 1.89e-3]),
                  1.0e-4: np.array([3.28e-3, 9.93e-2, 1.34e-2]),
                  3.0e-5: np.array([1.78e-6, 3.02e-3, 9.30e-1]),
                  1.0e-5: np.array([1.67e-6, 1.70e-6, 1.95e-6]),
                  3.0e-6: np.array([1.69e-6, 1.83e-6, 3.04e-6]),
                  1.0e-6: np.array([2.42e-6, 2.43e-6, 1.98e-6]),
                  3.0e-7: np.array([2.05e-6, 1.81e-6, 2.23e-6]),
                  1.0e-7: np.array([2.16e-6, 2.46e-6, 1.96e-6])},
        2.48e-7: {1.0e-3: np.array([4.53e-4, 3.68e-6, 1.75e-5]),
                  3.0e-4: np.array([1.54e-5, 8.91e-3, 3.53e-4]),
                  1.0e-4: np.array([2.00e-5, 1.47e-4, 5.19e-4]),
                  3.0e-5: np.array([1.72e-7, 3.52e-3, 1.34e-7]),
                  1.0e-5: np.array([1.72e-7, 1.63e-7, 1.61e-7]),
                  3.0e-6: np.array([2.05e-7, 2.61e-7, 3.47e-7]),
                  1.0e-6: np.array([2.47e-7, 2.54e-7, 4.29e-7]),
                  3.0e-7: np.array([1.89e-7, 2.08e-7, 2.05e-7]),
                  1.0e-7: np.array([2.14e-7, 2.03e-7, 1.94e-7])}}

if __name__ == "__main__":
    main()
