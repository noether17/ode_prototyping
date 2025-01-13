import argparse
import glob
import os
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import struct

from time import perf_counter

dim = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_pattern", "--fp", type=str,
                        help="pattern for globbing input files")
    args = parser.parse_args()
    file_pattern = args.file_pattern

    data_dict = {}

    for file_path in glob.glob(file_pattern):
        with open(file_path, 'rb') as input_file:
            n_times = int.from_bytes(input_file.read(8), 'little')
            n_var = int.from_bytes(input_file.read(8), 'little')
            N = int(n_var / 6)
            softening = struct.unpack('d', input_file.read(8))[0]
            tol = float(file_path.split('_tol_')[1].split('.bin')[0])

            if N not in data_dict:
                data_dict[N] = {}

            if softening not in data_dict[N]:
                data_dict[N][softening] = {}

            if tol not in data_dict[N][softening]:
                data_dict[N][softening][tol] = []

            # read the file into arrays
            times = np.empty(n_times, dtype=float)
            states = np.empty([n_times, n_var], dtype=float)
            for i in np.arange(n_times):
                times[i] = struct.unpack('d', input_file.read(8))[0]
                states[i] = [struct.unpack('d', input_file.read(8))[0]
                             for j in np.arange(n_var)]

            # compute energies for this run
            energies = compute_energies(states)

            fractional_dE = fractional_dE_vectorized(energies, energies[0])

            max_frac_dE = np.max(np.abs(fractional_dE))
            data_dict[N][softening][tol].append(max_frac_dE)
            print(f"N={N}, sof={softening}, tol={tol}, max dE/E={max_frac_dE}")

    print(data_dict)
    n_plot_cols = int(np.sqrt(len(data_dict)))
    n_plot_rows = int(np.ceil(len(data_dict) / n_plot_cols))
    print(f"n_plot_rows={n_plot_rows}, n_plot_cols={n_plot_cols}")
    fig, axs = plt.subplots(n_plot_rows, n_plot_cols, constrained_layout=True)
    enlargement_factor = 1.5
    fig.set_figheight(enlargement_factor * n_plot_rows * fig.get_figheight())
    fig.set_figwidth(enlargement_factor * n_plot_cols * fig.get_figwidth())
    for i, N in enumerate(sorted(data_dict)):
        print(f"i={i}, N={N}")
        if n_plot_cols > 1: plot_indices = int(i / n_plot_rows), i % n_plot_rows
        else: plot_indices = i
        print(f"plot_indices={plot_indices}")
        for softening in sorted(data_dict[N]):
            tols = sorted(data_dict[N][softening].keys())
            mean_frac_dEs = [np.mean(data_dict[N][softening][tol]) for tol in tols]
            axs[plot_indices].loglog(tols, mean_frac_dEs, label=f"{softening:.2e}")
        axs[plot_indices].set_xlabel("Tolerance")
        axs[plot_indices].set_ylabel(r"$\frac{E - E_0}{E_0}$")
        axs[plot_indices].set_title(f"N={N}")
        axs[plot_indices].legend()
    plt.show()
    plt.savefig(f"FractionalChangeInEnergyPlot.png")

# calculate potential energy for a single state. assumes all masses are 1.
@nb.njit()
def potential_energy(state_positions):
    N = int(state_positions.size / dim)
    energy = 0.0
    for i in np.arange(N):
        for j in np.arange(i + 1, N):
            pos_i = state_positions[i*dim:(i+1)*dim]
            pos_j = state_positions[j*dim:(j+1)*dim]
            dr = pos_j - pos_i
            energy -= 1.0 / np.sqrt(np.sum(dr*dr))
    return energy

# calculate kinetic energy for a single state. assumes all masses are 1.
@nb.njit()
def kinetic_energy(state_velocities):
    N = int(state_velocities.size / dim)
    energy = np.sum(state_velocities*state_velocities) / 2.0
    return energy

@nb.njit(parallel=True)
def compute_energies(states):
    print("Computing energies.")
    n_states = states.shape[0]
    offset = int(states.shape[1] / 2)
    energies = np.zeros(n_states)
    for i in nb.prange(n_states):
        energies[i] = potential_energy(states[i, :offset]) + \
                kinetic_energy(states[i, offset:])
    return energies

@nb.vectorize([nb.float64(nb.float64, nb.float64)], nopython=True)
def fractional_dE_vectorized(energies, energies_0):
    return (energies - energies_0) / energies_0

if __name__ == "__main__":
    main()
