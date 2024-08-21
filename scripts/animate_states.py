import argparse
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

animation_time = 10.0 # seconds to run animation
max_fps = 30 # maximum frames per second
dim = 3 # number of dimensions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename

    states = np.loadtxt(filename, delimiter=',', dtype=float)
    times = states[:, 0]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ani = anim.FuncAnimation(fig, plot_frame, times, fargs=(states, ax))

    frames_per_second = times.size / animation_time # TODO: limit to max_fps
    writer = anim.PillowWriter(fps=frames_per_second)
    N = int((states.shape[1] - 1) / 6)
    ani.save(f"animation_{N}_particles.gif", writer=writer)

def plot_frame(t, states, ax):
    row_index = np.where(states[:, 0] == t)
    N = int((states.shape[1] - 1) / 6)
    dof = N * dim
    current_frame = states[row_index, 1:dof+1].reshape((N, dim))
    ax.clear()
    for point in current_frame:
        ax.scatter(point[0], point[1], point[2], marker=',', color='k', alpha=0.2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    return ax

if __name__ == "__main__":
    main()
