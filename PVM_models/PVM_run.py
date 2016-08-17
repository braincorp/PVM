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
import multiprocessing as mp
import os
import logging
import argparse
import subprocess
import re
import string
import json
from PVM_models.PVM_Manager import Manager
import PVM_framework.CoreUtils as CoreUtils
import PVM_framework.PVM_Storage as PVM_Storage
import PVM_framework.PVM_Create as PVM_Create
import PVM_framework.PVM_datasets as PVM_datasets
import PVM_framework.PVM_SignalProvider as PVM_SignalProvider
import PVM_framework.PVM_options as PVM_options
import PVM_framework.PVM_display_helper as PVM_display_helper
from boto.utils import get_instance_metadata


def cleanup(s):
    ansi_escape = re.compile(r'\x1b[^m]*m')
    s = ansi_escape.sub('', s)
    s = s.strip()
    s = filter(lambda x: x in string.printable, s)
    return s


def run_model(evaluate=False,
              filename="",
              cores="",
              name="",
              description="",
              remote="",
              display=False,
              dataset="",
              meta={},
              options_given={},
              storage=None,
              port="9000",
              checkpoint=True,
              upgrade_only=False):
    """
    In this demo a future/predictive encoder is being instantiated to predict a camera image
    based on two previous frames. The system is built into a three layer hierarchy in which each next
    layer is predicting the hidden activations of the lower one.

    In addition the errors from each later are being backpropagated down to the previous layer.

    In addition to that, errors generated at context blocka are also being backpropagated to the originating
    unit. Consequently the error and signals flows in both directions through the entire system.
    """
    if options_given == {} and filename == "" and remote == "":
        logging.error("No options were given, don't know what to run! Try running with -h option.")
        exit()
    options = PVM_options.parse_options(options_given)
    if remote != "":
        filename = storage.get(remote)
        logging.info("Loaded a remote simulation dict %s" % remote)
    logging.info("Following options were given: %s" % json.dumps(options, sort_keys=True, indent=4))
    if os.path.isfile(filename):
        simulation_dict = CoreUtils.load_model(filename)
        if "options" in simulation_dict:
            options = PVM_options.parse_options(options_given, options_in_the_dict=simulation_dict['options'])
        else:
            options = PVM_options.parse_options(options_given)
        logging.info("Loaded the dictionary")
        if cores is not "":
            simulation_dict['num_proc'] = int(cores)
        else:
            simulation_dict['num_proc'] = min(2*mp.cpu_count()/3, simulation_dict["stage0_size"]/2)
        PVM_Create.upgrade(simulation_dict)
        PVM_Create.upgrade_dictionary_to_ver1_0(simulation_dict)
        logging.info("Running on %d cpu's" % simulation_dict['num_proc'])
    else:
        options = PVM_options.parse_options(options)
        simulation_dict = PVM_Create.generate_dict_options(name=name,
                                                           description=description,
                                                           options=options
                                                           )

        if cores is not "":
            simulation_dict['num_proc'] = int(cores)
        else:
            if options["model_type"] != "tiny":
                simulation_dict['num_proc'] = 2*mp.cpu_count()/3
            else:
                simulation_dict['num_proc'] = 1
        logging.info("Generated the dictionary")
    logging.info("Full set of options: %s" % json.dumps(options, sort_keys=True, indent=4))
    if options["new_name"] != "":
        simulation_dict['name'] = options["new_name"]
        options["new_name"] = ""
    if "disable_lateral" in options.keys() and options["disable_lateral"] == "1":
        simulation_dict["disable_lateral"] = True
    if "disable_feedback" in options.keys() and options["disable_feedback"] == "1":
        simulation_dict["disable_feedback"] = True
    if dataset == "":
        dataset = options["dataset"]
    else:
        options["dataset"] = dataset

    PVM_set = PVM_datasets.PVMDataset(dataset, storage=storage)
    PVM_Create.apply_options(simulation_dict, options)
    if upgrade_only:
        CoreUtils.save_model(simulation_dict, filename)
        to_folder = "PVM_models/%s_%s_%s" % (simulation_dict['timestamp'], simulation_dict['name'], simulation_dict['hash'])
        from_path = filename
        storage.put(from_path=from_path, to_folder=to_folder, overwrite=True)
        return

    if options['supervised'] == "1":
        logging.info("Running in the supervised mode")
        for (i, k) in enumerate(simulation_dict['learning_rates']):
            k[0] = 0
            logging.info("Setting learning rate %d to zero")
        simulation_dict['readout_learning_rate'][0] = float(options["supervised_rate"])
        logging.info("Setting additional_learning_rate to %f" % simulation_dict['readout_learning_rate'][0])

    if not evaluate:
        status_file = "/tmp/%s_%s_%s" % (simulation_dict['timestamp'], simulation_dict['name'], simulation_dict['hash'])
        f = open(status_file, "w")
        branch = subprocess.Popen('git rev-parse --abbrev-ref HEAD', shell=True, stdout=subprocess.PIPE).stdout.read()
        f.write("BRANCH=%s\n" % cleanup(branch))
        f.write("TIMESTAMP=%s\n" % simulation_dict['timestamp'])
        f.write("NAME=%s\n" % simulation_dict['name'])
        f.write("HASH=%s\n" % simulation_dict['hash'])
        f.write("DATASET=%s\n" % dataset)
        f.write("OPTIONS=%s\n" % json.dumps(options))
        f.close()

        remove_artifact_files = True
        if meta == {}:
            logging.info("Not running on an Amazon EC2 instance, apparently")
            logging.info("Not running on an Amazon EC2 instance, apparently: So, not automatically removing downloaded artifact files.")
            remove_artifact_files = False
        elif options['supervised'] != '1':
            host = meta['public-ipv4']
            logging.info("Running on amazon instance %s. Adding active job" % host)
            storage.put(from_path=status_file, to_folder='DARPA/active_jobs/', overwrite=True)
        # Train
        signal = PVM_SignalProvider.SimpleSignalProvider(files=PVM_set.training,
                                                         storage=storage,
                                                         frame_resolution=(simulation_dict['input_array'].shape[1], simulation_dict['input_array'].shape[0]),
                                                         heatmap_resolution=simulation_dict['readout_arrays'][0].shape[:2][::-1],
                                                         channel="default",
                                                         remove_files=remove_artifact_files,
                                                         reverse=(int(options['reverse']) > 0))
        manager = Manager(simulation_dict,
                          int(options['steps']),
                          signal_provider=signal,
                          record=False,
                          video_recorder=PVM_display_helper.VideoRecorder(rec_filename="PVM_recording.avi"),
                          do_display=display,
                          checkpoint=checkpoint,
                          checkpoint_storage=storage,
                          dataset_name=dataset)
        CoreUtils.run_model(simulation_dict, manager, port=int(port))

        if filename != "" and options['supervised'] != "1":
            CoreUtils.save_model(simulation_dict, filename)
            to_folder = "PVM_models/%s_%s_%s" % (simulation_dict['timestamp'], simulation_dict['name'], simulation_dict['hash'])
            from_path = filename
            storage.put(from_path=from_path, to_folder=to_folder, overwrite=True)
            if remove_artifact_files:
                os.remove(from_path)
        elif options['supervised'] == "1":
            CoreUtils.save_model(simulation_dict, "PVM_state_supervised_%s_%d_%d_%f.p.gz" % (dataset, simulation_dict['N'][0], int(options['steps']), float(options['supervised_rate'])))
            to_folder = "PVM_models/%s_%s_%s" % (simulation_dict['timestamp'], simulation_dict['name'], simulation_dict['hash'])
            from_path = "./PVM_state_supervised_%s_%d_%d_%f.p.gz" % (dataset, simulation_dict['N'][0], int(options['steps']), float(options['supervised_rate']))
            storage.put(from_path=from_path, to_folder=to_folder, overwrite=True)
            if remove_artifact_files:
                os.remove(from_path)
        else:
            CoreUtils.save_model(simulation_dict, "PVM_state_final.p.gz")
            to_folder = "PVM_models/%s_%s_%s" % (simulation_dict['timestamp'], simulation_dict['name'], simulation_dict['hash'])
            from_path = "./PVM_state_final.p.gz"
            storage.put(from_path=from_path, to_folder=to_folder, overwrite=True)
            if remove_artifact_files:
                os.remove(from_path)
    else:
        print "Evaluating the system"
        logging.info("Evaluating the system")
        to_folder = "PVM_models/%s_%s_%s/eval_%09d/" % (simulation_dict['timestamp'], simulation_dict['name'], simulation_dict['hash'], simulation_dict['N'])
        # Evaluate
        signal = PVM_SignalProvider.SimpleSignalProvider(files=PVM_set.testing,
                                                         storage=storage,
                                                         frame_resolution=(simulation_dict['input_array'].shape[1], simulation_dict['input_array'].shape[0]),
                                                         heatmap_resolution=simulation_dict['readout_array_float00'].shape[:2][::-1],
                                                         channel="default",
                                                         reverse=(int(options['reverse']) > 0))
        name = "PVM_train_eval_%s_%09d_test_combined.avi" % (simulation_dict['hash'], simulation_dict['N'])
        manager = Manager(simulation_dict,
                          steps_to_run=-1,
                          signal_provider=signal,
                          record=True,
                          video_recorder=PVM_display_helper.VideoRecorder(rec_filename=name),
                          do_display=display,
                          evaluate=True,
                          collect_error=True)
        manager.freeze_learning()
        CoreUtils.run_model(simulation_dict, manager, port=int(port))
        from_path = name
        storage.put(from_path=from_path, to_folder=to_folder, overwrite=True)
        os.remove(name)
        logging.info("Finished on test files")
        # Individual files
        for (i, test) in enumerate(PVM_set.all):
            print "Running on %s" % test
            logging.info("Running on %s" % test)
            name = "PVM_eval_%s_%09d_%01d_%s.avi" % (simulation_dict['hash'], simulation_dict['N'], i, test[1])
            signal = PVM_SignalProvider.SimpleSignalProvider(files=[test],
                                                             storage=storage,
                                                             frame_resolution=(simulation_dict['input_array'].shape[1], simulation_dict['input_array'].shape[0]),
                                                             heatmap_resolution=simulation_dict['readout_array_float00'].shape[:2][::-1],
                                                             channel="default")
            manager = Manager(simulation_dict,
                              steps_to_run=-1,
                              signal_provider=signal,
                              record=True,
                              video_recorder=PVM_display_helper.VideoRecorder(rec_filename=name),
                              do_display=display,
                              evaluate=True)
            CoreUtils.run_model(simulation_dict, manager, port=int(port))
            from_path = name
            storage.put(from_path=from_path, to_folder=to_folder, overwrite=True)
            os.remove(name)
            logging.info("Finished on %s" % test)


if __name__ == '__main__':
    logging.basicConfig(filename="PVM.log", level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(thread)d PVM_run : %(message)s ')
    logging.info("###################################################################")
    logging.info("                     STARTING NEW RUN                              ")
    logging.info("###################################################################")
    parser = argparse.ArgumentParser(description=PVM_options.get_option_help(), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-e", "--evaluate", help="Evaluate the trained system on a set of test movies", action="store_true")
    parser.add_argument("-B", "--backup", help="Store a simulation snapshot every 100.000 steps", action="store_true")
    parser.add_argument("-q", "--quiet", help="Print minimal logs to stdout", action="store_true")
    parser.add_argument("-D", "--display", help="Pull up display window", action="store_true")
    parser.add_argument("-u", "--upgrade_only", help="Load and upgrade the dictionary but don't run", action="store_true")
    parser.add_argument("-f", "--file", help="File to load", type=str, default="")
    parser.add_argument("-c", "--cores", help="Number of cores to use", type=str, default="")
    parser.add_argument("-d", "--description", help="Short description of the simulation", type=str, default="")
    parser.add_argument("-r", "--remote", help="Download and run a remote simulation", type=str, default="")
    parser.add_argument("-n", "--name", help="Short name of the simulation", type=str, default="PVM_simulation")
    parser.add_argument("-s", "--set", help="Name of the dataset", type=str, default="")
    parser.add_argument("-S", "--spec", help="Specification file name (file in .json format)", type=str, default="")
    parser.add_argument("-p", "--port", help="Debug console port. Access debug console by typing \"nc localhost _port_\"", type=str, default="9000")
    parser.add_argument('-O', '--options', type=json.loads, help="Option dictionary (as described above) given in json form '{\"key1\": \"value1\"}\'.", default='{}')
    Storage = PVM_Storage.Storage()
    try:
        meta = get_instance_metadata(timeout=2, num_retries=2)
    except:
        meta = {}
    args = parser.parse_args()
    have_display = args.display
    if not args.quiet:
        logging.getLogger().addHandler(logging.StreamHandler())
    try:
        from Tkinter import Tk
        Tk()
    except:
        have_display = False
        print "No display detected, turning off visualizations."

    using_cmdline_options = len(args.options) > 0
    using_json_options = len(args.spec) > 0
    assert not (using_cmdline_options and using_json_options), "We don't support using both command line and json spec options at the same time. (Suggestion: put all options in the json spec)"

    options = args.options
    if args.spec != "":
        options = json.load(open(args.spec, "r"))
    run_model(evaluate=args.evaluate,
              filename=args.file,
              cores=args.cores,
              description=args.description,
              display=have_display,
              name=args.name,
              remote=args.remote,
              dataset=args.set,
              meta=meta,
              options_given=options,
              storage=Storage,
              port=args.port,
              checkpoint=args.backup,
              upgrade_only=args.upgrade_only
              )
