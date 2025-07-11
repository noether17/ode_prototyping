import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

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
                [''.join(bm["name"].split('<')[1:]).split('>/')[0].split(', ')
                 for bm in benchmarks])
        system_sizes = np.array([int(tp_list[0]) for tp_list in template_params])
        int_methods = np.array([tp_list[1].split('BT')[1].split('>')[0]
                                for tp_list in template_params])
        par_methods = np.array([tp_list[2].split('Executor')[0]
                                for tp_list in template_params])
        items_per_second = np.array([float(bm["items_per_second"]) for bm in benchmarks])
        print(template_params)
        print(system_sizes)
        print(int_methods)
        print(par_methods)
        print(items_per_second)

        unique_int_methods = ['HE21', 'RKF45', 'DOPRI5', 'DVERK', 'RKF78']
        unique_par_methods = ['SingleThreaded', 'ThreadPool4>', 'ThreadPool8>', 'ThreadPool12>', 'ThreadPool16>', 'Cuda']
        for int_method in unique_int_methods:
            int_method_indices = np.where(int_methods == int_method)
            for par_method in unique_par_methods:
                par_method_indicies = np.where(par_methods == par_method)
                current_indices = np.intersect1d(int_method_indices, par_method_indicies)
                #print(f"Int: {int_method}; Par: {par_method}; N: {system_sizes[current_indices]}; IPS: {items_per_second[current_indices]}")
                plt.loglog(system_sizes[current_indices], items_per_second[current_indices], label=par_method)
            plt.xlabel("Number of Particles")
            plt.ylabel("Items per second")
            plt.title(f"Performance of {int_method}")
            plt.legend()
            plt.grid()
            plt.show()

if __name__ == "__main__":
    main()
