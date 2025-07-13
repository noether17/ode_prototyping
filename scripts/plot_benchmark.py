import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename

    with open(filename) as input_file:
        # read json file
        data = json.load(input_file)
        caches = data["context"]["caches"]
        benchmarks = data["benchmarks"]

        # parse data
        family_indices = np.array([int(bm["family_index"]) for bm in benchmarks])
        template_params = np.array(
                [bm["name"][bm["name"].find('<') + 1:
                            bm["name"].rfind('>')].split(', ')
                 for bm in benchmarks])
        data_type = template_params[0][-1] # assumes one data type per input file
        system_sizes = np.array([int(tp_list[0]) for tp_list in template_params])
        int_methods = np.array([tp_list[1].split('BT')[1].split('>')[0]
                                for tp_list in template_params])
        par_methods = np.array([
            tp_list[2].split('Executor')[0].replace('<', '-').replace('>', '')
            for tp_list in template_params])
        real_time = np.array([float(bm["real_time"]) for bm in benchmarks])
        nsq_per_rt = system_sizes**2 / real_time

        # plot data
        plot_benchmarks(int_methods, par_methods, system_sizes, nsq_per_rt, data_type)
        plot_benchmarks(par_methods, int_methods, system_sizes, nsq_per_rt, data_type)

def plot_benchmarks(outer_field, inner_field, x, y, data_type):
    unique_outer = unique_list(outer_field)
    unique_inner = unique_list(inner_field)
    for i_label in unique_outer:
        i_label_indices = np.where(outer_field == i_label)
        for j_label in unique_inner:
            j_label_indices = np.where(inner_field == j_label)
            current_indices = np.intersect1d(i_label_indices, j_label_indices)
            plt.loglog(x[current_indices], y[current_indices], label=j_label)
        plt.xlabel(r'$N$ (Number of Particles)')
        plt.ylabel(f'$N^2$ / s')
        plt.title(f"Performance of {i_label} with {data_type} elements")
        plt.legend()
        plt.grid()
        plt.show()

def unique_list(input_list):
    '''Get list of unique values that maintains order of first appearance.'''
    result = []
    result_items = set()
    for item in input_list:
        if item not in result_items:
            result_items.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    main()
