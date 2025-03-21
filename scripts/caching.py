import numpy as np
import os
import os.path
import struct

dim = 3

def compute_from_file(compute_function, result_name, state_filename):
    cache_filename = f"{result_name}_cache_{state_filename.split('.bin')[0]}.npy"
    if os.path.isfile(cache_filename) and \
            os.path.getmtime(cache_filename) > os.path.getmtime(state_filename):
                print(f"Reading cached {result_name} values for {state_filename}")
                return np.load(cache_filename)
    with open(state_filename, 'rb') as state_file:
        n_times = int.from_bytes(state_file.read(8), 'little')
        n_var = int.from_bytes(state_file.read(8), 'little')
        dof = int(n_var / 2) # degrees of freedom
        N = int(dof / dim)
        softening = struct.unpack('d', state_file.read(8))[0]

        # read the file into arrays
        times = np.empty(n_times, dtype=float)
        positions = np.empty([n_times, dof], dtype=float)
        velocities = np.empty([n_times, dof], dtype=float)
        for i in np.arange(n_times):
            times[i] = struct.unpack('d', state_file.read(8))[0]
            positions[i] = [struct.unpack('d', state_file.read(8))[0]
                            for j in np.arange(dof)]
            velocities[i] = [struct.unpack('d', state_file.read(8))[0]
                             for j in np.arange(dof)]

        # compute energies for this run
        results = compute_function(positions, velocities)
        results = np.vstack([times, results])
        results = results.transpose()
        np.save(cache_filename, results)
        return results
