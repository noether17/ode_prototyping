import argparse
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.interpolate as interp
import struct

from nbody_io import caching
from nbody_io import bin_reader as br
from nbody_physics import energy

dim = 3 # number of dimensions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename
    softening = br.read_metadata(filename)[-1]

    results = caching.compute_from_file(lambda positions, velocities:
                                        energy.compute_softened_energies(
                                            positions, velocities, softening),
                                        "softened_energy", filename)
    times = results[:, 0]
    energies = results[:, 1]

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

if __name__ == "__main__":
    main()
