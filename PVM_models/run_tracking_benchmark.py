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
import matplotlib
matplotlib.use('Agg')
import os
import logging
import PVM_framework.PVM_Storage as PVM_Storage
import argparse
from PVM_tools.benchmark import TrackerBenchmark
from PVM_models.PVM_tracker import PVMVisionTracker
import glob
import PVM_framework.PVM_datasets as PVM_datasets
from boto.utils import get_instance_metadata


if __name__ == "__main__":
    logging.basicConfig(filename="PVM_tracker.log", level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(thread)d PVM_tracker : %(message)s ')
    desc = ""
    have_display = True
    try:
        from Tkinter import Tk
        Tk()
    except:
        have_display = False
        print "No display detected, turning off visualizations."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-c", "--channel", type=str, default="default", help="Channel")
    parser.add_argument("-k", "--cores", type=str, default="4", help="NUmber of cores to execute on")
    parser.add_argument("-d", "--directory", type=str, default="DARPA/DATA/TrackerBenchmark", help="Directory indicating the dataset to un on")
    parser.add_argument("-f", "--filename", type=str, default="", help="Filename to run on")
    parser.add_argument("-r", "--remote", default="", help="Filename in the could")
    parser.add_argument("-e", "--execute", help="Actually run the trackers", action="store_true")
    parser.add_argument("-p", "--prefix", type=str, default="", help="Output directory prefix")
    parser.add_argument("-o", "--output", type=str, default="", help="Directory to either store or load from")
    parser.add_argument("-s", "--set", help="Name of the dataset", type=str, default="green_ball")
    parser.add_argument("-n", "--no_gt", help="Dont display ground truth", action="store_true")
    parser.add_argument("-R", "--resolution", help="Resolution ", type=str, default="96")
    parser.add_argument("-S", "--steps_per_frame", help="How many steps of the PVM before an estimate is made ", type=str, default="1")
    parser.add_argument("-T0", "--tracker_PVM", help="Use PVM tracker ", action="store_true")
    parser.add_argument("-T1", "--tracker_null", help="Use null tracker ", action="store_true")
    parser.add_argument("-T2", "--tracker_center", help="Use center tracker ", action="store_true")
    parser.add_argument("-T3", "--tracker_HS", help="Use HS color histogram tracker ", action="store_true")
    parser.add_argument("-T4", "--tracker_UV", help="Use UV color histogram tracker ", action="store_true")
    parser.add_argument("-T5", "--tracker_TLD", help="Use opentld tracker ", action="store_true")
    parser.add_argument("-T6", "--tracker_CMT", help="Use CMT tracker ", action="store_true")
    parser.add_argument("-T7", "--tracker_STRUCK", help="Use STRUCK tracker ", action="store_true")
    args = parser.parse_args()
    remove_files = True
    ts = PVM_Storage.Storage()
    try:
        meta = get_instance_metadata(timeout=2, num_retries=2)
    except:
        meta = {}
    if meta == {}:
        remove_files = False
    if args.output == "":
        output_dir = os.path.expanduser("~/benchmark_results/")+args.prefix
        output_dir = os.path.normpath(output_dir)
    else:
        output_dir = os.path.normpath(args.output)
    if args.set == "all":
        test_files_with_channel = []
        for ds in ("stop_sign", "green_ball", "face"):
            test_files_with_channel.extend(PVM_datasets.sets["%s_training" % ds])
    else:
        test_files_with_channel = PVM_datasets.sets["%s_testing" % args.set]
    test_files = []
    for (file, channel) in test_files_with_channel:
        test_files.append(file)
    if args.execute:
        trackers = []
        if args.tracker_null:
            from other_trackers.null_vision_tracker import NullVisionTracker
            trackers.append(NullVisionTracker())
        if args.tracker_center:
            from other_trackers.center_vision_tracker import CenterVisionTracker
            trackers.append(CenterVisionTracker())
        if args.tracker_CMT:
            from other_trackers.cmt_vision_tracker import CMTVisionTracker
            trackers.append(CMTVisionTracker())
        if args.tracker_TLD:
            from other_trackers.tld_vision_tracker import TLDVisionTracker
            trackers.append(TLDVisionTracker())
        if args.tracker_STRUCK:
            from other_trackers.struck_tracker import StruckTracker
            trackers.append(StruckTracker())
        if args.tracker_HS:
            from other_trackers.color_vision_tracker import HSHistogramBackprojectionTracker
            trackers.append(HSHistogramBackprojectionTracker())
        if args.tracker_UV:
            from other_trackers.color_vision_tracker import UVHistogramBackprojectionTracker
            trackers.append(UVHistogramBackprojectionTracker())
        if args.tracker_PVM:
            PVM_tracker = PVMVisionTracker(filename=args.filename, remote_filename=args.remote, cores=args.cores, storage=ts, steps_per_frame=int(args.steps_per_frame))
            trackers.append(PVM_tracker)
            output_dir = os.path.expanduser("~/benchmark_results/")+args.prefix+"_"+PVM_tracker.prop_dict["name"]+"_"+str(PVM_tracker.prop_dict["N"][0])
            output_dir = os.path.normpath(output_dir)
        TB = TrackerBenchmark(output_dir=output_dir,
                              resolution=(int(args.resolution), int(args.resolution)),
                              have_display=have_display,
                              draw_gt=not args.no_gt,
                              perturb_gt=0.4,
                              storage=ts)
        TB.run(test_files, trackers, ts, channel="default", remove_downloads=remove_files)
        TB.save_results()
    else:
        TB = TrackerBenchmark(output_dir=output_dir,
                              resolution=(int(args.resolution), int(args.resolution)),
                              have_display=have_display,
                              draw_gt=not args.no_gt,
                              perturb_gt=0.4,
                              storage=ts)
    if args.tracker_PVM:
        PVM_tracker.finish()
    TB.get_summary()
    TB.plot_all()
    benchmark_dir = output_dir
    base_dir = os.path.basename(os.path.normpath(benchmark_dir))
    if args.tracker_PVM:
        to_folder = "PVM_benchmark/%s_%s_%s/Benchmark_%09d_%s/" % (PVM_tracker.prop_dict['timestamp'], PVM_tracker.prop_dict['name'], PVM_tracker.prop_dict['hash'], PVM_tracker.prop_dict["N"], base_dir)
        logging.info("Uploading benchmark result to Storage from %s to %s" % (os.path.normpath(benchmark_dir), to_folder))
        for filename in glob.glob(os.path.normpath(benchmark_dir) + "/*"):
            if os.path.isfile(filename):
                logging.info("Inspecting file %s " % filename)
                ts.put(from_path=filename, to_folder=to_folder, overwrite=True)
        logging.info("Done, Uploading benchmark result to Amazon")
