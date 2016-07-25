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
import os
import logging
import argparse
import PVM_framework.CoreUtils as CoreUtils
import PVM_framework.PVM_Create as PVM_Create
import PVM_framework.PVM_Storage as PVM_Storage


if __name__ == '__main__':
    logging.basicConfig(filename="PVM_upgrade.log", level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(thread)d PVM_run : %(message)s ')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("###################################################################")
    logging.info("                     STARTING NEW RUN                              ")
    logging.info("###################################################################")
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File to load", type=str, default="")
    parser.add_argument("-r", "--remote", help="Download and run a remote simulation", type=str, default="")
    parser.add_argument("-d", "--destination", help="Where to save the model", type=str, default="PVM_models/")
    parser.add_argument("-n", "--name", help="New name", type=str, default="")
    args = parser.parse_args()
    Storage = PVM_Storage.Storage()
    if args.remote != "":
        filename = Storage.get(args.remote)
        logging.info("Loaded a remote simulation dict %s" % args.remote)
    if os.path.isfile(filename):
        simulation_dict = CoreUtils.load_model(filename)
        logging.info("Loaded the dictionary")

    PVM_Create.upgrade_dictionary_to_ver1_0(simulation_dict)
    for k in sorted(simulation_dict.keys()):
        print k
    if args.name != "":
        simulation_dict['name'] = args.name
    CoreUtils.save_model(simulation_dict, "PVM_failsafe_%010d.p.gz" % int(simulation_dict['N'][0]))
    to_folder = "PVM_models/%s_%s_%s" % (simulation_dict['timestamp'], simulation_dict['name'], simulation_dict['hash'])
    from_path = "./PVM_failsafe_%010d.p.gz" % int(simulation_dict['N'][0])
    logging.info("Uploading %s/%s" % (to_folder, from_path[2:]))
    Storage.put(from_path=from_path, to_folder=to_folder, overwrite=True)
