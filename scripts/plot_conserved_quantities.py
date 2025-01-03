import argparse
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.interpolate as interp
import struct

dim = 3 # number of dimensions

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
        data = np.array([[struct.unpack('d', data_file.read(8))[0]
                          for j in np.arange(n_var + 1)]
                         for i in np.arange(n_times)])
    times = data[:, 0]
    states = data[:, 1:-1]

    energies = compute_energies(states)

    # interpolate
    max_points = 1001
    if energies.size > max_points:
        print(f"Number of points greater than maximum ({max_points}). Interpolating.")
        spline = interp.make_interp_spline(times, energies, bc_type='natural')
        interp_times = np.linspace(times[0], times[-1], num=max_points)
        interp_energies = spline(interp_times)
    else:
        interp_times = times
        interp_energies = energies

    # plot
    plt.semilogy(interp_times[1:],
                 np.abs((interp_energies[1:] - energies[0]) / energies[0]))
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\frac{E(t) - E_0}{E_0}$")
    plt.title("Fractional Change in Energy over Time")
    plt.show()

# calculate potential energy for a single state. assumes all masses are 1.
@numba.njit()
def potential_energy(state_positions):
    N = int(state_positions.size / dim)
    softening = 0.01 / (N * (N - 1)) ###
    energy = 0.0
    for i in np.arange(N):
        for j in np.arange(i + 1, N):
            pos_i = state_positions[i*dim:(i+1)*dim]
            pos_j = state_positions[j*dim:(j+1)*dim]
            dr = np.linalg.norm(pos_j - pos_i)
            energy -= np.arctan(dr / softening) / softening
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
