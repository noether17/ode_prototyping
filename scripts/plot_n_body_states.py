import argparse
import numpy as np
import matplotlib.pyplot as plt

from nbody_io import bin_reader as br

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename

    states, softening = br.read_states(filename)
    dof = int((states.shape[1] - 1) / 2) # degrees of freedom
    times = states[:, 0]
    positions = states[:, 1:dof + 1]
    velocities = states[:, dof + 1:]
    n_particles = int((states.shape[1] - 1) / 6)
    for i in np.arange(n_particles):
        plt.plot(positions[:, 3 * i], positions[:, 3 * i + 1],
                 label=f"Particle {i}")
    plt.title(f"Trajectories for {filename}", wrap=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()
