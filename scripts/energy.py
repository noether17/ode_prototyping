import math
import numba as nb
import numba.cuda
import numpy as np
import time

dim = 3
threads_per_block = 256
min_blocks_per_grid = 1#28 # Numba gives a performance warning below this number.

#'''
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
#'''

def compute_potential_energies(positions):
    n_particles = int(positions.shape[1] / dim)
    n_pairs = int(n_particles * (n_particles - 1) / 2)
    print(f"Computing PE for {n_pairs} pairs of particles.")
    blocks_per_grid = int(math.ceil(n_pairs / threads_per_block))
    if False and nb.cuda.is_available() and blocks_per_grid >= min_blocks_per_grid:
        print(f"Using CUDA for PE calculation.")
        n_states = positions.shape[0]
        print(f"n_states == {n_states}")
        PEs = np.empty(n_states)
        dev_positions = nb.cuda.to_device(positions)
        dev_pair_potential_energies = nb.cuda.device_array([n_states, n_pairs])
        state_potential_energy_cuda[blocks_per_grid, threads_per_block] \
                (dev_pair_potential_energies, dev_positions)
        pair_potential_energies = dev_pair_potential_energies.copy_to_host()
        for i in np.arange(n_states):
            PEs[i] = np.sum(pair_potential_energies[i, :])
            print(f"PEs[{i}] == {PEs[i]}")
        #for i in np.arange(n_states):
        #    PEs[i] = nb.cuda.reduce(lambda a, b: a + b)(dev_pair_potential_energies[i, :])
        #    print(f"PEs[{i}] == {PEs[i]}")
        return PEs
    else:
        print(f"Using CPU for PE calculation.")
        return compute_potential_energies_parallel(positions)

@nb.njit(parallel=True)
def compute_potential_energies_parallel(positions):
    n_states = positions.shape[0]
    n_particles = int(positions.shape[1] / dim)
    PEs = np.zeros(n_states)
    for t in nb.prange(n_states):
        for i in np.arange(n_particles):
            for j in np.arange(i + 1, n_particles):
                pos_i = positions[t, i*dim:(i+1)*dim]
                pos_j = positions[t, j*dim:(j+1)*dim]
                dr = pos_j - pos_i
                PEs[t] -= 1.0 / np.sqrt(np.sum(dr*dr))
    return PEs

@nb.cuda.jit()
def state_potential_energy_cuda(pair_potential_energies, positions):
    pair_id = nb.cuda.threadIdx.x + nb.cuda.blockIdx.x * nb.cuda.blockDim.x
    if pair_id < pair_potential_energies.shape[1]:
        n_particles = int(positions.shape[1] / dim)
        n_minus_half = n_particles - 0.5
        i = int(n_minus_half - math.sqrt(n_minus_half * n_minus_half - 2.0 * pair_id))
        j = int(pair_id - (n_particles - 1) * i + (i * (i + 1)) / 2 + 1)
        state_id = 0
        while state_id < positions.shape[0]:
            pos_i = positions[state_id, i*dim:(i+1)*dim]
            pos_j = positions[state_id, j*dim:(j+1)*dim]
            dx = pos_j[0] - pos_i[0]
            dy = pos_j[1] - pos_i[1]
            dz = pos_j[2] - pos_i[2]
            pair_potential_energies[state_id, pair_id] = \
                    -1.0 / math.sqrt(dx*dx + dy*dy + dz*dz)
            state_id += 1

def compute_kinetic_energies(velocities):
    KEs = np.empty(velocities.shape[0])
    for i in np.arange(velocities.shape[0]):
        KEs[i] = kinetic_energy(velocities[i])
    return KEs

'''
def compute_energies(states):
    n_particles = int(states.shape[1] / (2 * dim))
    print(f"Number of particles: {n_particles}")
    n_pairs = int(n_particles * (n_particles - 1) / 2)
    pe_blocks_per_grid = int(math.ceil(n_pairs / threads_per_block))
    if nb.cuda.is_available() and pe_blocks_per_grid >= min_blocks_per_grid:
        print("Using CUDA.")
        return compute_energies_cuda(states)
    print("Using CPU.")
    return compute_energies_parallel(states)
#'''

@nb.njit(parallel=True)
def compute_energies_parallel(states):
    n_states = states.shape[0]
    energies = np.zeros(n_states)
    for i in nb.prange(n_states):
        energies[i] = potential_energy(states[i, :]) + \
                kinetic_energy(states[i, :])
    return energies

@nb.njit()
def potential_energy(state):
    n_particles = int(state.size / (2 * dim))
    energy = 0.0
    for i in np.arange(n_particles):
        for j in np.arange(i + 1, n_particles):
            pos_i = state[i*dim:(i+1)*dim]
            pos_j = state[j*dim:(j+1)*dim]
            dr = pos_j - pos_i
            energy -= 1.0 / np.sqrt(np.sum(dr*dr))
    return energy

@nb.njit()
def kinetic_energy(velocities):
    return np.sum(velocities*velocities) / 2.0

def compute_energies_cuda(states):
    n_states = states.shape[0]
    energies = np.zeros(n_states)

    n_particles = int(states.shape[1] / (2 * dim))
    n_pairs = int(n_particles * (n_particles - 1) / 2)
    pair_potential_energies = nb.cuda.mapped_array(n_pairs)

    vel_offset = int(states.shape[1] / 2)
    square_velocities = nb.cuda.mapped_array(vel_offset)

    pe_blocks_per_grid = int(math.ceil(n_pairs / threads_per_block))
    ke_blocks_per_grid = int(math.ceil(vel_offset / threads_per_block))
    for i in np.arange(n_states):
        potential_energy_cuda[pe_blocks_per_grid, threads_per_block] \
                (states[i], pair_potential_energies)
        kinetic_energy_cuda[ke_blocks_per_grid, threads_per_block] \
                (states[i], square_velocities)
        energies[i] = np.sum(pair_potential_energies) + np.sum(square_velocities) / 2.0
    return energies

@nb.cuda.jit()
def potential_energy_cuda(state, pair_potential_energies):
    pair_id = nb.cuda.threadIdx.x + nb.cuda.blockIdx.x * nb.cuda.blockDim.x
    if pair_id < pair_potential_energies.size:
        n_particles = int(state.size / (2 * dim))
        n_minus_half = n_particles - 0.5
        i = int(n_minus_half - math.sqrt(n_minus_half * n_minus_half - 2.0 * pair_id))
        j = int(pair_id - (n_particles - 1) * i + (i * (i + 1)) / 2 + 1)
        pos_i = state[i*dim:(i+1)*dim]
        pos_j = state[j*dim:(j+1)*dim]
        dx = pos_j[0] - pos_i[0]
        dy = pos_j[1] - pos_i[1]
        dz = pos_j[2] - pos_i[2]
        pair_potential_energies[pair_id] = -1.0 / math.sqrt(dx*dx + dy*dy + dz*dz)

@nb.cuda.jit()
def kinetic_energy_cuda(state, square_velocities):
    i = nb.cuda.threadIdx.x + nb.cuda.blockIdx.x * nb.cuda.blockDim.x
    vel_offset = int(state.size / 2)
    if i < vel_offset:
        square_velocities[i] = state[i + vel_offset] * state[i + vel_offset]
