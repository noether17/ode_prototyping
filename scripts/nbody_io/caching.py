import numpy as np
import os
import os.path
import struct

from . import bin_reader as br

dim = 3

def compute_from_file(compute_function, result_name, state_filename):
    cache_filename = f"{result_name}_cache_{state_filename.split('.bin')[0]}.npy"
    if os.path.isfile(cache_filename) and \
            os.path.getmtime(cache_filename) > os.path.getmtime(state_filename):
                print(f"Reading cached {result_name} values for {state_filename}")
                return np.load(cache_filename)
    states, softening = br.read_states(state_filename)
    dof = int((states.shape[1] - 1) / 2) # degrees of freedom
    times = np.ascontiguousarray(states[:, 0])
    positions = np.ascontiguousarray(states[:, 1:dof + 1])
    velocities = np.ascontiguousarray(states[:, dof + 1:])

    # compute energies for this run
    results = compute_function(positions, velocities)
    results = np.vstack([times, results])
    results = results.transpose()
    np.save(cache_filename, results)
    return results
