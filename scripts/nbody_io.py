import numpy as np
import struct

def read_binary_file_metadata(input_filename: str):
    with open(input_filename, mode='rb') as input_file:
        n_times = int.from_bytes(input_file.read(8), 'little')
        n_var = int.from_bytes(input_file.read(8), 'little')
        softening = struct.unpack('d', input_file.read(8))[0]
        return n_times, n_var, softening

def load_from_binary_file(input_filename: str):
    with open(input_filename, mode='rb') as input_file:
        n_times = int.from_bytes(input_file.read(8), 'little')
        n_var = int.from_bytes(input_file.read(8), 'little')
        softening = struct.unpack('d', input_file.read(8))[0]
        states = np.fromfile(input_file)
        states = states.reshape(n_times, n_var + 1)
        return states, softening
