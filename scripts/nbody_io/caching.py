import numpy as np
import os
import os.path
import struct

from . import bin_reader as br

dim = 3

def compute_from_file(compute_function, result_name, state_filename):
    # Create cache directory.
    cache_dir = f"{result_name}_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Set cache file path.
    cache_filename = f"{result_name}_cache_{state_filename.split('.bin')[0]}.npy"
    cache_filepath = os.path.join(cache_dir, cache_filename)

    # Read cached results, if cache file exists.
    if os.path.isfile(cache_filepath) and \
            os.path.getmtime(cache_filepath) > os.path.getmtime(state_filename):
                print(f"Reading cached {result_name} values for {state_filename}")
                return np.load(cache_filepath)

    # Read state file.
    states, softening = br.read_states(state_filename)
    dof = int((states.shape[1] - 1) / 2) # degrees of freedom
    times = np.ascontiguousarray(states[:, 0])
    positions = np.ascontiguousarray(states[:, 1:dof + 1])
    velocities = np.ascontiguousarray(states[:, dof + 1:])

    # Compute results and save cache file.
    results = compute_function(positions, velocities)
    results = np.vstack([times, results])
    results = results.transpose()
    np.save(cache_filepath, results)
    return results
