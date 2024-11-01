import numpy as np
import matplotlib.pyplot as plt

def main():
    cpu_data = np.loadtxt("RKF78_n_body_output.txt", delimiter=',')
    cpu_times = cpu_data[:, 0]
    state_offset = int((cpu_data.shape[1] - 1) / 2)
    cpu_pos = cpu_data[:, 1:1+state_offset]
    cpu_vel = cpu_data[:, 1+state_offset:]
    n_particles = int((cpu_data.shape[1] - 1) / 6)
    for i in np.arange(n_particles):
        plt.plot(cpu_pos[:, 3*i], cpu_pos[:, 3*i + 1], label=f"Particle {i}")
    plt.title("CPU Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()

    cpu_parallel_data = np.loadtxt("RKF78_parallel_n_body_output.txt", delimiter=',')
    cpu_parallel_times = cpu_parallel_data[:, 0]
    state_offset = int((cpu_parallel_data.shape[1] - 1) / 2)
    cpu_parallel_pos = cpu_parallel_data[:, 1:1+state_offset]
    cpu_parallel_vel = cpu_parallel_data[:, 1+state_offset:]
    n_particles = int((cpu_parallel_data.shape[1] - 1) / 6)
    for i in np.arange(n_particles):
        plt.plot(cpu_parallel_pos[:, 3*i], cpu_parallel_pos[:, 3*i + 1], label=f"Particle {i}")
    plt.title("CPU Parallel Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()

    cuda_data = np.loadtxt("n_body_output.txt", delimiter=',')
    cuda_times = cuda_data[:, 0]
    state_offset = int((cuda_data.shape[1] - 1) / 2)
    cuda_pos = cuda_data[:, 1:1+state_offset]
    cuda_vel = cuda_data[:, 1+state_offset:]
    n_particles = int((cpu_data.shape[1] - 1) / 6)
    for i in np.arange(n_particles):
        plt.plot(cuda_pos[:, 3*i], cuda_pos[:, 3*i + 1], label=f"Particle {i}")
    plt.title("CUDA Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()
