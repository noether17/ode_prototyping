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
        unique_int_methods = unique_list(int_methods)
        unique_par_methods = unique_list(par_methods)
        for int_method in unique_int_methods:
            int_method_indices = np.where(int_methods == int_method)
            for par_method in unique_par_methods:
                par_method_indicies = np.where(par_methods == par_method)
                current_indices = np.intersect1d(int_method_indices, par_method_indicies)
                plt.loglog(system_sizes[current_indices], nsq_per_rt[current_indices], label=par_method)
            plt.xlabel(r'$N$ (Number of Particles)')
            plt.ylabel(r'$N^2$ / s')
            plt.title(f"Performance of {int_method} with {data_type} elements")
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
