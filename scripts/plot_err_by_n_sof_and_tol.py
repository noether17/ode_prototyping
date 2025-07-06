import argparse
import glob
import os
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import struct

from nbody_io import caching
from nbody_physics import energy

dim = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_pattern", "--fp", type=str,
                        help="pattern for globbing input files")
    args = parser.parse_args()
    file_pattern = args.file_pattern

    data_dict = {}

    for filename in glob.glob(file_pattern):
        N = int(filename.split('_')[1])
        softening = float(filename.split('_sof_')[1].split('_')[0])
        tolerance = float(filename.split('_tol_')[1].split('.bin')[0])

        if N not in data_dict:
            data_dict[N] = {}

        if softening not in data_dict[N]:
            data_dict[N][softening] = {}

        if tolerance not in data_dict[N][softening]:
            data_dict[N][softening][tolerance] = []

        print(f"Getting results for N={N}, sof={softening}, tol={tolerance}")
        #results = caching.compute_from_file(energy.compute_energies, "energy",
        #                                    filename)
        results = caching.compute_from_file(lambda positions, velocities:
                                            energy.compute_softened_energies(
                                                positions, velocities, softening),
                                            "softened_energy", filename)
        times = results[:, 0]
        energies = results[:, 1]
        fractional_dE = fractional_dE_vectorized(energies, energies[0])

        max_frac_dE = np.max(np.abs(fractional_dE))
        data_dict[N][softening][tolerance].append(max_frac_dE)
        print(f"N={N}, sof={softening}, tol={tolerance}, max dE/E={max_frac_dE}")

    n_plot_cols = int(np.sqrt(len(data_dict)))
    n_plot_rows = int(np.ceil(len(data_dict) / n_plot_cols))
    fig, axs = plt.subplots(n_plot_rows, n_plot_cols, constrained_layout=True)
    enlargement_factor = 1.5
    fig.set_figheight(enlargement_factor * n_plot_rows * fig.get_figheight())
    fig.set_figwidth(enlargement_factor * n_plot_cols * fig.get_figwidth())
    for i, N in enumerate(sorted(data_dict)):
        if n_plot_cols > 1: plot_indices = int(i / n_plot_cols), i % n_plot_cols
        else: plot_indices = i
        for softening in sorted(data_dict[N]):
            tols = sorted(data_dict[N][softening].keys())
            mean_frac_dEs = [np.mean(data_dict[N][softening][tol]) for tol in tols]
            axs[plot_indices].loglog(tols, mean_frac_dEs, label=f"{softening:.2e}")
        axs[plot_indices].set_xlabel("Tolerance")
        axs[plot_indices].set_ylabel(r"$\frac{E - E_0}{E_0}$")
        axs[plot_indices].set_title(f"N={N}")
        axs[plot_indices].legend()
    plt.savefig(f"FractionalChangeInEnergyPlot.png")
    plt.show()

@nb.vectorize([nb.float64(nb.float64, nb.float64)], nopython=True)
def fractional_dE_vectorized(energies, energies_0):
    return (energies - energies_0) / energies_0

if __name__ == "__main__":
    main()
