import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import struct

dim = 3 # number of dimensions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename

    with open(filename, mode='rb') as data_file:
        n_times = int.from_bytes(data_file.read(8), 'little')
        n_var = int.from_bytes(data_file.read(8), 'little')
        states = np.array([[struct.unpack('d', data_file.read(8))[0]
                            for j in np.arange(n_var + 1)]
                           for i in np.arange(n_times)])
    times = states[:, 0]
    N = int((states.shape[1] - 1) / (2 * dim))
    positions = states[:, 1:1+N*dim]
    velocities = states[:, 1+N*dim:-1]

    energies = [potential_energy(pos) + kinetic_energy(vel)
                for pos, vel in zip(positions, velocities)]
    plt.semilogy(times[1:], np.abs((energies[1:] - energies[0]) / energies[0]))
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\frac{E(t) - E_0}{E_0}$")
    plt.title("Fractional Change in Energy over Time")
    plt.show()

# calculate potential energy for a single state. assumes all masses are 1.
def potential_energy(state_positions):
    N = int(state_positions.size / dim)
    energy = 0.0
    for i in np.arange(N):
        for j in np.arange(i + 1, N):
            pos_i = state_positions[i*dim:(i+1)*dim]
            pos_j = state_positions[j*dim:(j+1)*dim]
            energy -= 1.0 / np.linalg.norm(pos_j - pos_i)
    return energy

# calculate kinetic energy for a single state. assumes all masses are 1.
def kinetic_energy(state_velocities):
    N = int(state_velocities.size / dim)
    energy = np.sum(state_velocities*state_velocities) / 2.0
    return energy

if __name__ == "__main__":
    main()
