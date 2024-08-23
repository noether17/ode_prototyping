import argparse
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp

animation_time = 10.0 # seconds to run animation
max_fps = 30 # maximum frames per second
dim = 3 # number of dimensions

# For updating progress
total_frames = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename

    # load and parse data
    states = np.loadtxt(filename, delimiter=',', dtype=float)
    times = states[:, 0]
    N = int((states.shape[1] - 1) / 6)
    dof = N * dim
    positions = states[:, 1:dof+1]

    #interpolate
    spline = interp.make_interp_spline(times, positions, bc_type='natural')
    global total_frames
    total_frames = int(max_fps * animation_time)
    interp_times = np.linspace(times[0], times[-1], num=total_frames)
    interp_positions = spline(interp_times)

    # animate
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ani = anim.FuncAnimation(fig, plot_frame, total_frames,
                             fargs=(interp_times, interp_positions, ax))
    writer = anim.PillowWriter(fps=max_fps)
    ani.save(f"animation_{N}_particles.gif", writer=writer)
    print() # newline after progress report

def plot_frame(frame_idx, times, positions, ax):
    t = times[frame_idx]
    current_frame = positions[frame_idx]
    N = int(current_frame.size / dim)
    ax.clear()
    for i in np.arange(N):
        ax.scatter(current_frame[i * dim], current_frame[i * dim + 1],
                   current_frame[i * dim + 2], marker=',', color='k', alpha=0.2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_title(f"Number of Particles: {N}\nt = {t:.2f}")

    global total_frames
    progress_percent = ((frame_idx + 1) / total_frames) * 100.0
    print(f"\rPlotted {progress_percent:.2f}% of frames.", end='')

    return ax

if __name__ == "__main__":
    main()
