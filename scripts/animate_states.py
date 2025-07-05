import argparse
import math
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
import struct

import nbody_io

animation_time = 30.0 # seconds to run animation
max_fps = 30 # maximum frames per second
dim = 3 # number of dimensions

# For updating progress
total_frames = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename
    method_str = method_str_from_filename(filename)

    # load and parse data
    states, softening = nbody_io.load_from_binary_file(filename)
    times = states[:, 0]
    N = int((states.shape[1] - 1) / 6)
    dof = N * dim
    positions = states[:, 1:dof+1]

    #interpolate
    spline = interp.make_interp_spline(times, positions, bc_type='natural')
    global total_frames
    total_frames = int(max_fps * animation_time) + 1
    interp_times = np.linspace(times[0], times[-1], num=total_frames)
    interp_positions = spline(interp_times)

    # animate
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    marker_size = (72.0 / fig.dpi)**2
    ani = anim.FuncAnimation(fig, plot_frame, total_frames,
                             fargs=(interp_times, interp_positions, method_str,
                                    ax, marker_size))
    writer = anim.PillowWriter(fps=max_fps)
    ani.save(f"animation_{filename}.gif", writer=writer)
    print() # newline after progress report

def plot_frame(frame_idx, times, positions, method_str, ax, marker_size):
    t = times[frame_idx]
    current_frame = positions[frame_idx]
    N = int(current_frame.size / dim)
    ax.clear()
    ax.scatter(current_frame[::dim], current_frame[1::dim],
               current_frame[2::dim], marker=',', color='k', alpha=1.0,
               s=marker_size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title(f"Method: {method_str}\nNumber of Particles: {N}\nt = {t:.2e}")

    global total_frames
    progress_percent = ((frame_idx + 1) / total_frames) * 100.0
    print(f"\rPlotted {progress_percent:.2f}% of frames.", end='')

    return ax

def method_str_from_filename(filename: str):
    if 'RKF78' in filename:
        return 'RKF78'
    return 'Unknown'

if __name__ == "__main__":
    main()
