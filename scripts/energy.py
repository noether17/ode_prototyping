import math
import numba as nb
import numba.cuda
import numpy as np
import time

dim = 3

# These parameters have been tuned for optimal performance on an RTX 4070 Mobile
# GPU. Incidentally, the min and max correspond to the same number of blocks.
threads_per_block = 64
min_blocks_per_grid = 1024 # Numba gives a performance warning below 128, but
                           # testing suggests the minimum should be higher.
max_cuda_threads = int(2**16) # Prevent device arrays from getting too large.
                              # Numba's reduce function seems to be inefficient,
                              # so reduction is performed on the CPU. Larger
                              # device arrays mean more data being transferred
                              # back to the CPU and less work being done on the
                              # GPU.

def compute_energies(positions, velocities):
    start = time.time()
    PEs = compute_potential_energies(positions)
    print(f"Time to compute potential energies: {time.time() - start}s")

    start = time.time()
    KEs = compute_kinetic_energies(velocities)
    print(f"Time to compute kinetic energies: {time.time() - start}s")

    start = time.time()
    energies = PEs + KEs
    print(f"Time to combine energies: {time.time() - start}s")
    return energies

def compute_softened_energies(positions, velocities, softening):
    start = time.time()
    PEs = compute_softened_potential_energies(positions, softening)
    print(f"Time to compute potential energies: {time.time() - start}s")

    start = time.time()
    KEs = compute_kinetic_energies(velocities)
    print(f"Time to compute kinetic energies: {time.time() - start}s")

    start = time.time()
    energies = PEs + KEs
    print(f"Time to combine energies: {time.time() - start}s")
    return energies

def compute_potential_energies(positions):
    if nb.cuda.is_available():
        n_particles = int(positions.shape[1] / dim)
        n_pairs = int(n_particles * (n_particles - 1) / 2)
        n_threads = min(n_pairs, max_cuda_threads)
        blocks_per_grid = int(math.ceil(n_threads / threads_per_block))
        if blocks_per_grid >= min_blocks_per_grid:
            print(f"Computing potential energies using CUDA.")
            return compute_potential_energies_cuda(positions)
    print(f"Computing potential energies using CPU.")
    return compute_potential_energies_parallel(positions)

def compute_potential_energies_cuda(positions):
    dev_positions = nb.cuda.to_device(positions)

    n_particles = int(positions.shape[1] / dim)
    n_pairs = int(n_particles * (n_particles - 1) / 2)
    n_threads = min(n_pairs, max_cuda_threads)
    blocks_per_grid = int(math.ceil(n_threads / threads_per_block))

    n_states = positions.shape[0]
    dev_pair_potential_energies = nb.cuda.device_array([n_states, n_threads])
    pair_potential_energy_kernel[blocks_per_grid, threads_per_block] \
            (dev_pair_potential_energies, dev_positions, n_pairs)
    pair_potential_energies = dev_pair_potential_energies.copy_to_host()
    return np.array([np.sum(pair_potential_energies[i, :])
                     for i in np.arange(n_states)])

@nb.cuda.jit()
def pair_potential_energy_kernel(pair_potential_energies, positions, n_pairs):
    thread_id = nb.cuda.threadIdx.x + nb.cuda.blockIdx.x * nb.cuda.blockDim.x
    state_id = 0
    n_particles = int(positions.shape[1] / dim)
    n_minus_half = n_particles - 0.5
    while state_id < positions.shape[0]:
        pair_id = thread_id
        thread_pe = 0.0
        while pair_id < n_pairs:
            i = int(n_minus_half -
                    math.sqrt(n_minus_half * n_minus_half - 2.0 * pair_id))
            j = int(pair_id - (n_particles - 1) * i + (i * (i + 1)) / 2 + 1)
            pos_i = positions[state_id, i*dim:(i+1)*dim]
            pos_j = positions[state_id, j*dim:(j+1)*dim]
            dx = pos_j[0] - pos_i[0]
            dy = pos_j[1] - pos_i[1]
            dz = pos_j[2] - pos_i[2]
            thread_pe -= 1.0 / math.sqrt(dx*dx + dy*dy + dz*dz)

            pair_id += nb.cuda.blockDim.x * nb.cuda.gridDim.x

        if thread_id < pair_potential_energies.shape[1]:
            pair_potential_energies[state_id, thread_id] = thread_pe

        state_id += 1

@nb.njit(parallel=True)
def compute_potential_energies_parallel(positions):
    n_states = positions.shape[0]
    n_particles = int(positions.shape[1] / dim)
    PEs = np.zeros(n_states)
    for state_id in nb.prange(n_states):
        for i in np.arange(n_particles):
            for j in np.arange(i + 1, n_particles):
                pos_i = positions[state_id, i*dim:(i+1)*dim]
                pos_j = positions[state_id, j*dim:(j+1)*dim]
                dr = pos_j - pos_i
                PEs[state_id] -= 1.0 / np.sqrt(np.sum(dr*dr))
    return PEs

def compute_softened_potential_energies(positions, softening):
    if nb.cuda.is_available():
        n_particles = int(positions.shape[1] / dim)
        n_pairs = int(n_particles * (n_particles - 1) / 2)
        n_threads = min(n_pairs, max_cuda_threads)
        blocks_per_grid = int(math.ceil(n_threads / threads_per_block))
        if blocks_per_grid >= min_blocks_per_grid:
            print(f"Computing softened potential energies using CUDA.")
            return compute_softened_potential_energies_cuda(positions, softening)
    print(f"Computing softened potential energies using CPU.")
    return compute_softened_potential_energies_parallel(positions, softening)

def compute_softened_potential_energies_cuda(positions, softening):
    dev_positions = nb.cuda.to_device(positions)

    n_particles = int(positions.shape[1] / dim)
    n_pairs = int(n_particles * (n_particles - 1) / 2)
    n_threads = min(n_pairs, max_cuda_threads)
    blocks_per_grid = int(math.ceil(n_threads / threads_per_block))

    n_states = positions.shape[0]
    dev_pair_potential_energies = nb.cuda.device_array([n_states, n_threads])
    pair_softened_potential_energy_kernel[blocks_per_grid, threads_per_block] \
            (dev_pair_potential_energies, dev_positions, n_pairs, softening)
    pair_potential_energies = dev_pair_potential_energies.copy_to_host()
    return np.array([np.sum(pair_potential_energies[i, :])
                     for i in np.arange(n_states)])

@nb.cuda.jit()
def pair_softened_potential_energy_kernel(pair_potential_energies, positions,
                                          n_pairs, softening):
    thread_id = nb.cuda.threadIdx.x + nb.cuda.blockIdx.x * nb.cuda.blockDim.x
    state_id = 0
    n_particles = int(positions.shape[1] / dim)
    n_minus_half = n_particles - 0.5
    while state_id < positions.shape[0]:
        pair_id = thread_id
        thread_pe = 0.0
        while pair_id < n_pairs:
            i = int(n_minus_half -
                    math.sqrt(n_minus_half * n_minus_half - 2.0 * pair_id))
            j = int(pair_id - (n_particles - 1) * i + (i * (i + 1)) / 2 + 1)
            pos_i = positions[state_id, i*dim:(i+1)*dim]
            pos_j = positions[state_id, j*dim:(j+1)*dim]
            dx = pos_j[0] - pos_i[0]
            dy = pos_j[1] - pos_i[1]
            dz = pos_j[2] - pos_i[2]
            thread_pe -= 1.0 / math.sqrt(dx*dx + dy*dy + dz*dz +
                                         softening*softening)

            pair_id += nb.cuda.blockDim.x * nb.cuda.gridDim.x

        if thread_id < pair_potential_energies.shape[1]:
            pair_potential_energies[state_id, thread_id] = thread_pe

        state_id += 1

@nb.njit(parallel=True)
def compute_softened_potential_energies_parallel(positions, softening):
    n_states = positions.shape[0]
    n_particles = int(positions.shape[1] / dim)
    PEs = np.zeros(n_states)
    for state_id in nb.prange(n_states):
        for i in np.arange(n_particles):
            for j in np.arange(i + 1, n_particles):
                pos_i = positions[state_id, i*dim:(i+1)*dim]
                pos_j = positions[state_id, j*dim:(j+1)*dim]
                dr = pos_j - pos_i
                PEs[state_id] -= 1.0 / np.sqrt(np.sum(dr*dr) +
                                               softening*softening)
    return PEs

def compute_kinetic_energies(velocities):
    return np.array([np.sum(state_velocities*state_velocities) / 2.0
                     for state_velocities in velocities])
