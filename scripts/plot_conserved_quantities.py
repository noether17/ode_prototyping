import argparse
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.interpolate as interp
import struct

dim = 3 # number of dimensions
global softening

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename

    # read input file
    with open(filename, mode='rb') as data_file:
        print("Reading input file.")
        n_times = int.from_bytes(data_file.read(8), 'little')
        n_var = int.from_bytes(data_file.read(8), 'little')
        global softening
        softening = struct.unpack('d', data_file.read(8))[0]
        print(softening)

        times = np.empty(n_times, dtype=float)
        states = np.empty([n_times, n_var], dtype=float)
        for i in np.arange(n_times):
            times[i] = struct.unpack('d', data_file.read(8))[0]
            states[i] = [struct.unpack('d', data_file.read(8))[0]
                         for j in np.arange(n_var)]

    energies = compute_energies(states)

    # interpolate
    max_points = 1001
    if energies.size > max_points:
        print(f"Number of points ({energies.size}) is greater than maximum " +
              f"({max_points}). Interpolating.")
        spline = interp.make_interp_spline(times, energies, bc_type='natural')
        interp_times = np.linspace(times[0], times[-1], num=max_points)
        interp_energies = spline(interp_times)
    else:
        interp_times = times
        interp_energies = energies

    fractional_dE = np.abs((interp_energies[1:] - energies[0]) / energies[0])
    print(f"Average fractional change in energy: {np.mean(fractional_dE)}")
    print(f"Initial energy: {energies[0]}")
    print(f"Mean energy: {np.mean(energies)}")

    # plot
    plt.semilogy(interp_times[1:], fractional_dE)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\frac{E(t) - E_0}{E_0}$")
    plt.title("Fractional Change in Energy over Time")
    plt.show()

# calculate potential energy for a single state. assumes all masses are 1.
@numba.njit()
def potential_energy(state_positions):
    global softening
    N = int(state_positions.size / dim)
    energy = 0.0
    for i in np.arange(N):
        for j in np.arange(i + 1, N):
            pos_i = state_positions[i*dim:(i+1)*dim]
            pos_j = state_positions[j*dim:(j+1)*dim]
            dr = pos_j - pos_i
            energy -= 1.0 / np.sqrt(np.sum(dr*dr) + softening*softening)
    return energy

# calculate kinetic energy for a single state. assumes all masses are 1.
@numba.njit()
def kinetic_energy(state_velocities):
    N = int(state_velocities.size / dim)
    energy = np.sum(state_velocities*state_velocities) / 2.0
    return energy

@numba.njit(parallel=True)
def compute_energies(states):
    print("Computing energies.")
    n_states = states.shape[0]
    offset = int(states.shape[1] / 2)
    energies = np.zeros(n_states)
    for i in numba.prange(n_states):
        energies[i] = potential_energy(states[i, :offset]) + \
                kinetic_energy(states[i, offset:])
    return energies

if __name__ == "__main__":
    main()
