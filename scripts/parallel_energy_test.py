import argparse
import glob

from nbody_io import caching
from nbody_physics import energy

dim = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_pattern", "-fp", type=str,
                        help="pattern for globbing input files")
    args = parser.parse_args()
    file_pattern = args.file_pattern

    data_dict = {}

    for filename in glob.glob(file_pattern):
        N = int(filename.split('_')[1])
        softening = float(filename.split('_sof_')[1].split('_')[0])
        tol = float(filename.split('_tol_')[1].split('.bin')[0])

        if N not in data_dict:
            data_dict[N] = {}

        if softening not in data_dict[N]:
            data_dict[N][softening] = {}

        if tol not in data_dict[N][softening]:
            data_dict[N][softening][tol] = []

        energies = caching.compute_from_file(energy.compute_energies, "energy",
                                             filename)

if __name__ == "__main__":
    main()
