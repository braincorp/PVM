# ==================================================================================
# Copyright (c) 2016, Brain Corporation
#
# This software is released under Creative Commons
# Attribution-NonCommercial-ShareAlike 3.0 (BY-NC-SA) license.
# Full text available here in LICENSE.TXT file as well as:
# https://creativecommons.org/licenses/by-nc-sa/3.0/us/legalcode
#
# In summary - you are free to:
#
#    Share - copy and redistribute the material in any medium or format
#    Adapt - remix, transform, and build upon the material
#
# The licensor cannot revoke these freedoms as long as you follow the license terms.
#
# Under the following terms:
#    * Attribution - You must give appropriate credit, provide a link to the
#                    license, and indicate if changes were made. You may do so
#                    in any reasonable manner, but not in any way that suggests
#                    the licensor endorses you or your use.
#    * NonCommercial - You may not use the material for commercial purposes.
#    * ShareAlike - If you remix, transform, or build upon the material, you
#                   must distribute your contributions under the same license
#                   as the original.
#    * No additional restrictions - You may not apply legal terms or technological
#                                   measures that legally restrict others from
#                                   doing anything the license permits.
# ==================================================================================
import argparse
from PVM_framework.PVM_Storage import Storage
import PVM_framework.CoreUtils as CoreUtils
import numpy as np
import matplotlib.pyplot as plt


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


def plot_weight_dists(simulation_dict):
    all_weights_s = np.array([])
    all_weights_i = np.array([])
    all_weights_d = np.array([])
    all_weights_e = np.array([])
    all_weights_c = np.array([])
    for u in simulation_dict["stage0"]:
        w = u['MLP_parameters']['weights'][0].copy()
        w_s = np.array([])
        w_d = np.array([])
        w_i = np.array([])
        w_e = np.array([])
        w_c = np.array([])
        m = 0
        for (i, b) in enumerate(u['signal_blocks']):
            n = np.prod(b[0].shape)
            w_s = np.hstack((w_s, w[i*4*n:i*4*n+n].flatten()))
            w_d = np.hstack((w_d, w[i*4*n+n:i*4*n+2*n].flatten()))
            w_i = np.hstack((w_i, w[i*4*n+2*n:i*4*n+3*n].flatten()))
            w_e = np.hstack((w_e, w[i*4*n+3*n:i*4*n+4*n].flatten()))
            m += 4*n
        if 'complex' in u.keys() and u['complex']:
            m = 2*np.prod(u['output_block'].shape)
            w = u['MLP_parameters']['weights'][1].copy()
            for (i, b) in enumerate(u['context_blocks']):
                n = np.prod(b[0].shape)
                w_c = np.hstack((w_c, w[m+i*n:m+i*n+n].flatten()))
        else:
            for (i, b) in enumerate(u['context_blocks']):
                n = np.prod(b[0].shape)
                w_c = np.hstack((w_c, w[m+i*n:m+i*n+n].flatten()))

        all_weights_s = np.hstack((all_weights_s, w_s))
        all_weights_i = np.hstack((all_weights_i, w_i))
        all_weights_d = np.hstack((all_weights_d, w_d))
        all_weights_e = np.hstack((all_weights_e, w_e))
        all_weights_c = np.hstack((all_weights_c, w_c))
    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    all_weights = [all_weights_s, all_weights_i, all_weights_d, all_weights_e, all_weights_c]
    plt.hist(all_weights, 15, normed=0, color=['red', 'yellow', 'green', 'gray', 'blue'], label=['Signal', 'Integral', 'Derivative', 'Error', 'Context'])
    plt.grid(True)
    plt.title("Weight distribution " + simulation_dict['name'])
    plt.legend(prop={'size': 9})
    plt.subplot(122)
    plt.hist(map(lambda x: np.abs(x), all_weights), 15, normed=0, color=['red', 'yellow', 'green', 'gray', 'blue'], label=['Signal', 'Integral', 'Derivative', 'Error', 'Context'])
    plt.grid(True)
    plt.title("Weight distribution (abs) " + simulation_dict['name'])
    plt.legend(prop={'size': 9})

    pdf_file = "PVM_%s_%s_weight_dist.pdf" % (simulation_dict['name'], simulation_dict['hash'])
    plt.savefig(pdf_file)
    return pdf_file


def plot_model(filename, remote, compare, display):
    ts = Storage()
    if remote != "":
        filename = ts.get(remote)
    if compare != "":
        filename1 = ts.get(compare)
        simulation_dict1 = CoreUtils.load_model(filename1)
    simulation_dict = CoreUtils.load_model(filename)

    num_layers = len(simulation_dict["state_arrays"])
    plt.figure(figsize=(7, 5))
    colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
    ids = np.where(simulation_dict['error_log'][0, :] > 0)[0]
    window = (ids.shape[0])/200
    small_window = (ids.shape[0])/40
    for r in range(num_layers):
        # plt.suptitle("MSE %s" % (simulation_dict['name']))
        plt.plot(simulation_dict['error_log'][0, ids][:-small_window], runningMeanFast(simulation_dict['error_log'][r+1, ids], small_window)[:-small_window], lw=1, c=colors[r % 7], label="%s l. %d" % (simulation_dict['name'], r))
    if compare:
        for r in range(num_layers):
            plt.plot(simulation_dict1['error_log'][0, ids][:-small_window], runningMeanFast(simulation_dict1['error_log'][r+1, ids], small_window)[:-small_window], linestyle="--", lw=1, c=colors[r % 7], label="%s l. %d" % (simulation_dict1['name'], r))

    plt.xlabel("Training time")
    plt.ylabel("MSE (averaged in %d step bins)" % (1000*small_window))
    plt.title("Learning curve (MSE) - individual layers")
    plt.grid(True)
    plt.legend(prop={'size': 9})
    pdf_file = "PVM_%s_%s.pdf" % (simulation_dict['name'], simulation_dict['hash'])
    if compare != "":
        pdf_file = "PVM_%s_%s_comp_%s.pdf" % (simulation_dict['name'], simulation_dict['hash'], simulation_dict1['name'])
    plt.savefig(pdf_file)
    to_folder = "DARPA/Simulations/%s_%s_%s/" % (simulation_dict['timestamp'], simulation_dict['name'], simulation_dict['hash'])
    ts.put(from_path=pdf_file, to_folder=to_folder, overwrite=True)

    plt.figure(figsize=(7, 5))
    colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
    ids = np.where(simulation_dict['error_log'][0, :] > 0)[0]
    summed = np.zeros_like(simulation_dict['error_log'][1, ids])
    for r in range(num_layers):
        summed += simulation_dict['error_log'][r+1, ids]
    plt.plot(simulation_dict['error_log'][0, ids][:-small_window], runningMeanFast(summed, small_window)[:-small_window], lw=1, c='r', label="MSE All layers, model %s" % simulation_dict['name'])
    if compare != "":
        summed = np.zeros_like(simulation_dict1['error_log'][1, ids])
        for r in range(num_layers):
            summed += simulation_dict1['error_log'][r+1, ids]
        plt.plot(simulation_dict1['error_log'][0, ids][:-small_window], runningMeanFast(summed, small_window)[:-small_window], lw=1, c='b', label="MSE All layers, model %s" % simulation_dict1['name'])

    plt.xlabel("Training time")
    plt.ylabel("MSE (averaged in %d step bins)" % (1000*small_window))
    plt.title("Learning curve (MSE) - whole system")
    plt.grid(True)
    plt.legend(prop={'size': 9})
    pdf_file = "PVM_%s_%s_summed.pdf" % (simulation_dict['name'], simulation_dict['hash'])
    if compare != "":
        pdf_file = "PVM_%s_%s_summed_comp_%s.pdf" % (simulation_dict['name'], simulation_dict['hash'], simulation_dict1['name'])
    plt.savefig(pdf_file)
    to_folder = "DARPA/Simulations/%s_%s_%s/" % (simulation_dict['timestamp'], simulation_dict['name'], simulation_dict['hash'])
    ts.put(from_path=pdf_file, to_folder=to_folder, overwrite=True)
    # pdf_file = plot_weight_dists(simulation_dict)
    # ts.put(from_path=pdf_file, to_folder=to_folder, overwrite=True)
    # if compare != "":
    # pdf_file = plot_weight_dists(simulation_dict1)
    # ts.put(from_path=pdf_file, to_folder=to_folder, overwrite=True)

    if display:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File to load", type=str, default="")
    parser.add_argument("-r", "--remote", help="Download and run a remote simulation", type=str, default="")
    parser.add_argument("-c", "--compare", help="Compare with this file", type=str, default="")
    parser.add_argument("-D", "--display", help="Pull up display window", action="store_true")
    args = parser.parse_args()
    plot_model(filename=args.file,
               remote=args.remote,
               compare=args.compare,
               display=args.display)
