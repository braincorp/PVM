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


"""
This module contains the TrackerBenchmark class which encapsulates all the aspects of a tracker benchmark.

Import this module as:
::

    import tracker_benchmark.benchmark

or:
::

    from tracker_benchmark import TrackerBenchmark

"""
import matplotlib
matplotlib.use('Agg') 
import PVM_tools.labeled_movie as lm
import PVM_framework.PVM_Storage as PVM_Storage
import argparse
from PVM_tools.bounding_region import BoundingRegion
import cv2
import traceback
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import os
import errno
import time
from matplotlib.backends.backend_pdf import PdfPages
import PVM_framework.PVM_datasets as tracker_datasets


class TrackerBenchmark(object):

    def __init__(self, output_dir="~/benchmark_results", resolution=None, have_display=True, draw_gt=True, perturb_gt=0, storage=None):
        """
        :param output_dir: directory where the benchmarks will be stored on the local machine
        :param resolution: desired resolution of the benchmark
        :param have_display: should the display window be pulled
        :param draw_gt: should the ground truth bounding box be drawn on output videos
        :param perturb_gt: should there be a perturbed grond truth tracker added for baseline comparisons. If zero no tracker
        is added, if greater than zero it sets the fraction of perturbation as a fraction of ground truith box size. Values
        near 0.4 are reasonable.
        :return: The benchmark object
        """
        self._raw_results = {}
        self._timing_results = {}
        self._processed_results = {}
        self._summary_results = {}
        self.draw_gt = draw_gt
        self.perturb_gt = perturb_gt
        self._colors = [(0, 0, 255),
                        (0, 255, 255),
                        (255, 255, 0),
                        (255, 0, 0),
                        (0, 255, 0),
                        (255, 0, 255),
                        (0, 0, 128),
                        (0, 128, 0),
                        (128, 0, 0),
                        (128, 255, 0),
                        (0, 255, 128),
                        (0, 128, 255),
                        (255, 128, 0),
                        (255, 0, 128),
                        (128, 0, 255)
                        ]
        self._styles = [{'c': 'r', 'l': '-'},
                        {'c': 'g', 'l': '-'},
                        {'c': 'b', 'l': '-'},
                        {'c': 'k', 'l': '-'},
                        {'c': 'c', 'l': '-'},
                        {'c': 'm', 'l': '-'},
                        {'c': 'y', 'l': '-'},
                        {'c': 'b', 'l': '-.'},
                        {'c': 'r', 'l': '-.'},
                        {'c': 'g', 'l': '-.'},
                        {'c': 'k', 'l': '-.'},
                        {'c': 'c', 'l': '-.'},
                        {'c': 'm', 'l': '-.'},
                        {'c': 'y', 'l': '-.'},
                        {'c': 'b', 'l': '--'},
                        {'c': 'r', 'l': '--'},
                        {'c': 'g', 'l': '--'},
                        {'c': 'k', 'l': '--'},
                        {'c': 'c', 'l': '--'},
                        {'c': 'm', 'l': '--'},
                        {'c': 'y', 'l': '--'},
                        {'c': 'b', 'l': ':'},
                        {'c': 'r', 'l': ':'},
                        {'c': 'g', 'l': ':'},
                        {'c': 'k', 'l': ':'},
                        {'c': 'c', 'l': ':'},
                        {'c': 'm', 'l': ':'},
                        {'c': 'y', 'l': ':'}
                        ]
        self.resolution = resolution
        self._video = None
        self.have_display = have_display
        try:
            os.makedirs(os.path.expanduser(output_dir))
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise
        self._output_dir = os.path.expanduser(output_dir)
        if os.path.isfile(self._output_dir+"/raw_data.p"):
            if os.path.isfile(self._output_dir+"/timing_data.p"):
                self.load_results(filename=self._output_dir+"/raw_data.p", timing_filename=self._output_dir+"/timing_data.p")
            else:
                self.load_results(filename=self._output_dir+"/raw_data.p")
        self.storage = storage

    def evaluate_on_file(self,
                         filepath,
                         trackers,
                         channel="default",
                         targets=None,
                         window=True,
                         save_movie=True):
        """
        Evaluates a given list of trackers on a given set of targets on a given channel. If target list is not given
        the trackers will be evaluated on all the targets available in the file. If a channel is not given, default
        channel will be used.

        :param filepath: path to the labeled movie file being evaluated
        :param trackers: list of tracker objects
        :param channel: channel to be evaluated ("default")
        :param window: show the window
        :param save_movie: save an avi file with the tracker performance.
        :return: new_results, False when all trackers have already been run on this file, otherwise True
        :rtype: bool
        """
        for tracker in trackers:
            tracker.reset()
        fc = lm.FrameCollection()
        fc.load_from_file(filepath)
        name = os.path.basename(filepath)
        all_trackers = trackers

        if targets is None:
            targets = fc.Frame(0).get_targets()
        for target in targets:
            if name not in self._raw_results.keys():
                self._raw_results[name] = {}
                self._timing_results[name] = {}

            if target not in self._raw_results[name].keys():
                self._raw_results[name][target] = {}
                self._timing_results[name][target] = {}

            need_to_run_trackers = []
            for tracker in trackers:
                if tracker.name not in self._raw_results[name][target]:
                    need_to_run_trackers.append(tracker)

            if len(need_to_run_trackers) == 0:
                print "Skipping", name, target, "because it has already been evaluated for all trackers."
                return False  # no new results

            if trackers != need_to_run_trackers:
                print "Warning, running only a subset of trackers can result in a different number of frames primed per run."
            trackers = need_to_run_trackers

            save_video_file = self._output_dir + "/" + name[:-4]+"_" + target + ".avi"
            if save_movie:
                self._video = cv2.VideoWriter()
                fps = 20
                im = fc.Frame(0).get_image(channel=channel)
                if self.resolution is not None:
                    im = cv2.resize(im, dsize=self.resolution)
                retval = self._video.open(os.path.expanduser(save_video_file),
                                          cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),
                                          fps, (im.shape[1], im.shape[0]))
                assert(retval)

            self._raw_results[name][target]["ground_truth"] = []
            if self.perturb_gt>0:
                self._raw_results[name][target]['GT_pert_%2.2f' % self.perturb_gt] = []
            for tracker in trackers:
                self._raw_results[name][target][tracker.name] = []
                self._timing_results[name][target][tracker.name] = {}
                self._timing_results[name][target][tracker.name]['abs'] = []
                self._timing_results[name][target][tracker.name]['norm'] = []
                tracker.reset()
            primed = False
            for i in xrange(len(fc)):
                img = fc.Frame(i).get_image(channel=channel)
                if self.resolution is not None:
                    old_shape = img.shape
                    img = cv2.resize(img, dsize=self.resolution)
                pixels = np.prod(img.shape[0:2])
                br = fc.Frame(i).get_label(channel=channel, target=target)
                if self.resolution is not None:
                    br.scale_to_new_image_shape(new_image_shape=self.resolution, old_image_shape=old_shape)
                else:
                    br.set_image_shape(img.shape)
                if not primed and br.empty:
                    self._raw_results[name][target]['ground_truth'].append(BoundingRegion())
                    for tracker in trackers:
                        self._raw_results[name][target][tracker.name].append(BoundingRegion())
                    if self.perturb_gt > 0:
                        self._raw_results[name][target]['GT_pert_%2.2f' % self.perturb_gt].append(BoundingRegion())
                    continue

                if not primed:
                    for tracker in trackers:
                        any_except = False
                        try:
                            tracker.prime(img, br.copy())
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except:
                            any_except = True
                            exctype, value = sys.exc_info()[:2]
                            if str(value).lower().find("prim") == -1:
                                print "Caught Exception during priming"
                                traceback.print_exc(file=sys.stdout)
                            else:
                                print "Caught Exception during priming", value
                            
                    if any_except:
                        self._raw_results[name][target]['ground_truth'].append(br)
                        for tracker in trackers:
                            self._raw_results[name][target][tracker.name].append(br)
                        continue
                    else:
                        primed = True

                disp_img = img.copy()
                self._raw_results[name][target]['ground_truth'].append(br)
                if self.perturb_gt>0:
                    brp = br.copy()
                    if not brp.empty:
                        brp.scale(1.0+self.perturb_gt*0.5*(1.0-np.random.rand()))
                        box = np.array(brp.get_box_pixels())
                        x_shift = int(box[2]*self.perturb_gt*0.5*(1.0-np.random.rand()))
                        y_shift = int(box[3]*self.perturb_gt*0.5*(1.0-np.random.rand()))
                        if self.resolution is not None:
                            shape = self.resolution
                        else:
                            shape=img.shape
                        box[0] = np.clip(box[0]+x_shift, 0, shape[0])
                        box[1] = np.clip(box[1]+y_shift, 0, shape[1])
                        box[2] = np.clip(box[2]+x_shift, 0, shape[0])
                        box[3] = np.clip(box[3]+y_shift, 0, shape[1])
                        pgt = BoundingRegion(image_shape=self.resolution, box=box)
                    else:
                        pgt = BoundingRegion()
                    self._raw_results[name][target]['GT_pert_%2.2f' % self.perturb_gt].append(pgt)

                for (j, tracker) in enumerate(trackers):
                    t_ = time.time()  # Performance timing
                    try:
                        tbr = tracker.track(img)
                        self._raw_results[name][target][tracker.name].append(tbr)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        print "Exception in the tracker code! " + tracker.name, 'frame', i
                        traceback.print_exc(file=sys.stdout)
                        self._raw_results[name][target][tracker.name].append(BoundingRegion())
                    t = time.time()  # Performance timing
                    self._timing_results[name][target][tracker.name]['abs'].append(t-t_)
                    self._timing_results[name][target][tracker.name]['norm'].append((t-t_)/pixels)
                if window or save_movie:
                    for (j, tracker) in enumerate(all_trackers):
                        tbr = self._raw_results[name][target][tracker.name][i]
                        tbr.draw_box(disp_img, color=self._colors[(j+1) % len(self._colors)], thickness=1, annotation=tracker.name, linetype=cv2.CV_AA)
                    if self.draw_gt:
                        br.draw_box(disp_img, color=self._colors[0], thickness=1, annotation="Ground truth", linetype=cv2.CV_AA)
                        if self.perturb_gt>0:
                            pgt.draw_box(disp_img, color=self._colors[-1], thickness=1, annotation="Perturbed truth", linetype=cv2.CV_AA)
                    if window and self.have_display:
                        cv2.imshow("Tracking", disp_img)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            break
                    if save_movie:
                        self._video.write(disp_img)
            if self.have_display and window:
                cv2.destroyAllWindows()
            if save_movie:
                self._video.release()
        return True  # new results

    def save_results(self):
        """
        :param filename: name of the file to be saved
        """
        file = open(self._output_dir + "/raw_data.p", "wb")
        cPickle.dump(self._raw_results, file, protocol=-1)
        file.close()
        file = open(self._output_dir + "/timing_data.p", "wb")
        cPickle.dump(self._timing_results, file, protocol=-1)
        file.close()

    def load_results(self, filename, timing_filename=None):
        """
        :param filename: path to the file loaded
        """
        file = open(filename, "rb")
        self._raw_results = cPickle.load(file)
        file.close()
        if timing_filename is not None:
            file = open(timing_filename, "rb")
            self._timing_results = cPickle.load(file)
            file.close()
        self.process_results()

    def process_results(self, filename=None):
        """
        Converts raw results into a more processed form

        :return: processed results, same indexing but other information
        """
        print "Processing called with filename " + str(filename)
        sys.stdout.flush()
        if filename is None:
            filenames = self._raw_results.keys()
        else:
            filenames = [filename]
        for filename in filenames:
            if filename not in self._processed_results.keys():
                self._processed_results[filename] = {}
            targets = self._raw_results[filename].keys()
            for target in targets:
                if target not in self._processed_results[filename].keys():
                    self._processed_results[filename][target] = {}

                tracker_names = copy.copy(self._raw_results[filename][target].keys())
                num_of_frames = len(self._raw_results[filename][target]["ground_truth"])
                for tracker in tracker_names:
                    center_dists_pix = np.zeros(num_of_frames,)-0.00000001
                    center_dists_rel = np.zeros(num_of_frames,)-0.00000001
                    bounding_box_dists_rel = np.zeros((num_of_frames, 2))+np.Inf
                    overlap = np.zeros(num_of_frames,)
                    presence = np.zeros(num_of_frames,)
                    target_present_and_tracked_frames = 0
                    target_absent_and_tracked_frames = 0
                    target_present_and_not_tracked_frames = 0
                    target_absent_and_not_tracked_frames = 0
                    for i in xrange(num_of_frames):
                        if not self._raw_results[filename][target]["ground_truth"][i].empty and \
                                not self._raw_results[filename][target][tracker][i].empty:
                            target_present_and_tracked_frames += 1
                            center_target = self._raw_results[filename][target]["ground_truth"][i].get_box_center_pixels()
                            center_target_rel = self._raw_results[filename][target]["ground_truth"][i].get_box_center_relative()
                            center_tracker = self._raw_results[filename][target][tracker][i].get_box_center_pixels()
                            center_tracker_rel = self._raw_results[filename][target][tracker][i].get_box_center_relative()
                            distance = np.sqrt((center_target[0]-center_tracker[0])**2 + (center_target[1]-center_tracker[1])**2)
                            center_dists_pix[i] = distance
                            distance_rel = np.sqrt((center_target_rel[0]-center_tracker_rel[0])**2 +
                                                   (center_target_rel[1]-center_tracker_rel[1])**2)
                            center_dists_rel[i] = distance_rel

                            intersection = self._raw_results[filename][target]["ground_truth"][i].get_box_intersection(self._raw_results[filename][target][tracker][i])
                            overlap[i] = (intersection.get_area_pixels()*1.0 /
                                          (self._raw_results[filename][target]["ground_truth"][i].get_area_pixels() +
                                           self._raw_results[filename][target][tracker][i].get_area_pixels() -
                                           intersection.get_area_pixels() + 0.00001))
                                           
                            bb = self._raw_results[filename][target]["ground_truth"][i].get_box_pixels()
                            shape_factor = self._raw_results[filename][target]["ground_truth"][i].image_shape_factor
                            if shape_factor is None:  # provide a reasonable default if shape_factor is not defined
                                shape_factor = (1000, 1000)
                            # bounding box width and height can be 0, which actually means a width of 1 pixel
                            # shape_factor is the shape of the original image used to compute the bounding box
                            # so we add 1.0/shape_factor to compensate and avoid division by 0
                            bounding_box_dists_rel[i, 0] = (center_target[0]-center_tracker[0])/float(bb[2]+1.0/shape_factor[0])
                            bounding_box_dists_rel[i, 1] = (center_target[1]-center_tracker[1])/float(bb[3]+1.0/shape_factor[1])

                        elif self._raw_results[filename][target]["ground_truth"][i].empty and \
                                not self._raw_results[filename][target][tracker][i].empty:
                            target_absent_and_tracked_frames += 1
                            presence[i] = 1
                        elif not self._raw_results[filename][target]["ground_truth"][i].empty and \
                                self._raw_results[filename][target][tracker][i].empty:
                            target_present_and_not_tracked_frames += 1
                            presence[i] = 2
                        else:
                            target_absent_and_not_tracked_frames += 1
                            presence[i] = 3
                            bounding_box_dists_rel[i, :] = 0
                    true_positives = (target_present_and_tracked_frames * 100.0/num_of_frames)
                    true_negatives = (target_absent_and_not_tracked_frames * 100.0/num_of_frames)
                    false_positives = (target_absent_and_tracked_frames * 100.0/num_of_frames)
                    false_negatives = (target_present_and_not_tracked_frames * 100.0/num_of_frames)
                    self._processed_results[filename][target][tracker] = {}
                    self._processed_results[filename][target][tracker]['total_frames'] = num_of_frames
                    self._processed_results[filename][target][tracker]['total_true_positive'] = target_present_and_tracked_frames
                    self._processed_results[filename][target][tracker]['total_true_negative'] = target_absent_and_not_tracked_frames
                    self._processed_results[filename][target][tracker]['total_false_positive'] = target_absent_and_tracked_frames
                    self._processed_results[filename][target][tracker]['total_false_negative'] = target_present_and_not_tracked_frames
                    self._processed_results[filename][target][tracker]['true_positive'] = true_positives
                    self._processed_results[filename][target][tracker]['true_negative'] = true_negatives
                    self._processed_results[filename][target][tracker]['false_positive'] = false_positives
                    self._processed_results[filename][target][tracker]['false_negative'] = false_negatives
                    self._processed_results[filename][target][tracker]['presence_score'] = ((true_negatives+true_positives-false_negatives-false_positives + 100)/2)
                    self._processed_results[filename][target][tracker]['distance_pix'] = center_dists_pix
                    self._processed_results[filename][target][tracker]['distance_rel'] = center_dists_rel
                    self._processed_results[filename][target][tracker]['bounding_box_dists_rel'] = bounding_box_dists_rel
                    self._processed_results[filename][target][tracker]['overlap'] = overlap
                    self._processed_results[filename][target][tracker]['presence'] = presence
                    if target_present_and_tracked_frames>0:
                        self._processed_results[filename][target][tracker]['avg_distance_pix'] = np.sum(center_dists_pix)/target_present_and_tracked_frames
                        self._processed_results[filename][target][tracker]['avg_distance_rel'] = np.sum(center_dists_rel)/target_present_and_tracked_frames
                    else:
                        self._processed_results[filename][target][tracker]['avg_distance_pix'] = 1000000000.0
                        self._processed_results[filename][target][tracker]['avg_distance_rel'] = 1000000000.0
        af = "all_files"
        at = "all_targets"
        if af not in self._processed_results.keys():
            self._processed_results[af] = {}
            self._processed_results[af][at] = {}
        
        sys.stdout.flush()
        # The following bit is quite simple but ugly, needs to be abstracted probably.
        for filename in filenames:
            for target in self._processed_results[filename]:
                for tracker in self._processed_results[filename][target]:
                    if tracker not in self._processed_results[af][at]:
                        self._processed_results[af][at][tracker] = {}
                        self._processed_results[af][at][tracker]['total_frames'] = 0
                        self._processed_results[af][at][tracker]['total_true_positive'] = 0
                        self._processed_results[af][at][tracker]['total_true_negative'] = 0
                        self._processed_results[af][at][tracker]['total_false_positive'] = 0
                        self._processed_results[af][at][tracker]['total_false_negative'] = 0
                        self._processed_results[af][at][tracker]['avg_distance_pix'] = 0
                        self._processed_results[af][at][tracker]['avg_distance_rel'] = 0
                        self._processed_results[af][at][tracker]['num_movies'] = 0
                        self._processed_results[af][at][tracker]['distance_pix'] = np.array([])
                        self._processed_results[af][at][tracker]['distance_rel'] = np.array([])
                        self._processed_results[af][at][tracker]['overlap'] = np.array([])
                        self._processed_results[af][at][tracker]['presence'] = np.array([])
                        self._processed_results[af][at][tracker]['bounding_box_dists_rel'] = np.zeros((0, 2))

                    N = self._processed_results[af][at][tracker]['num_movies']
                    self._processed_results[af][at][tracker]['total_frames'] += self._processed_results[filename][target][tracker]['total_frames']
                    self._processed_results[af][at][tracker]['total_true_positive'] += self._processed_results[filename][target][tracker]['total_true_positive']
                    self._processed_results[af][at][tracker]['total_true_negative'] += self._processed_results[filename][target][tracker]['total_true_negative']
                    self._processed_results[af][at][tracker]['total_false_positive'] += self._processed_results[filename][target][tracker]['total_false_positive']
                    self._processed_results[af][at][tracker]['total_false_negative'] += self._processed_results[filename][target][tracker]['total_false_negative']

                    self._processed_results[af][at][tracker]['true_positive'] = (self._processed_results[af][at][tracker]['total_true_positive'] *
                                                                                 100.0 / self._processed_results[af][at][tracker]['total_frames'])
                    self._processed_results[af][at][tracker]['true_negative'] = (self._processed_results[af][at][tracker]['total_true_negative'] *
                                                                                 100.0 / self._processed_results[af][at][tracker]['total_frames'])
                    self._processed_results[af][at][tracker]['false_positive'] = (self._processed_results[af][at][tracker]['total_false_positive'] *
                                                                                  100.0 / self._processed_results[af][at][tracker]['total_frames'])
                    self._processed_results[af][at][tracker]['false_negative'] = (self._processed_results[af][at][tracker]['total_false_negative'] *
                                                                                  100.0 / self._processed_results[af][at][tracker]['total_frames'])
                    self._processed_results[af][at][tracker]['presence_score'] = ((self._processed_results[af][at][tracker]['true_positive'] +
                                                                                   self._processed_results[af][at][tracker]['true_negative'] -
                                                                                   self._processed_results[af][at][tracker]['false_positive'] -
                                                                                   self._processed_results[af][at][tracker]['false_negative'] + 100) / 2
                                                                                  )
                    self._processed_results[af][at][tracker]['avg_distance_pix'] = (self._processed_results[af][at][tracker]['avg_distance_pix'] *
                                                                                    N+self._processed_results[filename][target][tracker]['avg_distance_pix'])/(N+1)
                    self._processed_results[af][at][tracker]['avg_distance_rel'] = (self._processed_results[af][at][tracker]['avg_distance_rel'] *
                                                                                    N+self._processed_results[filename][target][tracker]['avg_distance_rel'])/(N+1)
                    self._processed_results[af][at][tracker]['num_movies'] = N+1
                    self._processed_results[af][at][tracker]['distance_pix'] = np.append(self._processed_results[af][at][tracker]['distance_pix'],
                                                                                         self._processed_results[filename][target][tracker]['distance_pix'])
                    self._processed_results[af][at][tracker]['distance_rel'] = np.append(self._processed_results[af][at][tracker]['distance_rel'],
                                                                                         self._processed_results[filename][target][tracker]['distance_rel'])
                    self._processed_results[af][at][tracker]['overlap'] = np.append(self._processed_results[af][at][tracker]['overlap'],
                                                                                    self._processed_results[filename][target][tracker]['overlap'])
                    self._processed_results[af][at][tracker]['presence'] = np.append(self._processed_results[af][at][tracker]['presence'],
                                                                                     self._processed_results[filename][target][tracker]['presence'])
                    self._processed_results[af][at][tracker]['bounding_box_dists_rel'] = \
                        np.concatenate((self._processed_results[af][at][tracker]['bounding_box_dists_rel'],
                                        self._processed_results[filename][target][tracker]['bounding_box_dists_rel']), axis=0)

        return self._processed_results

    def calc_precision(self, filename, target, tracker, thresholds, precision_level, absolute):
        data = self._processed_results[filename][target][tracker]['distance_pix']
        data = data[np.where(data >= 0)]
        precision = np.zeros(thresholds.shape)
        for j, x in enumerate(thresholds):
            if absolute:
                precision[j] = 1.0*np.where(data < x)[0].shape[0]/self._processed_results[filename][target]["ground_truth"]['total_true_positive']
            else:
                if self._processed_results[filename][target][tracker]['total_true_positive']>0:
                    precision[j] = 1.0*np.where(data < x)[0].shape[0]/self._processed_results[filename][target][tracker]['total_true_positive']
                else:
                    precision[j] = 0
        return precision, precision[np.argmin(np.abs(thresholds-precision_level))]

    def plot_precision(self, plot_individual_passes=False, precision_level=20, absolute=True):
        """
        Plots the precision plots as described by [TRBEN13]_.

        .. [TRBEN13] Online Object Tracking: A Benchmark, Yi Wu and Jongwoo Lim and Ming-Hsuan Yang,
            IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013

        :return:
        """
        if plot_individual_passes:
            names = self._processed_results.keys()
        else:
            names = ["all_files"]
        thresholds = np.arange(51)
        for filename in names:
            for target in self._processed_results[filename].keys():
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111)
                trackers = sorted(self._processed_results[filename][target].keys())
                precisions_at_level = np.zeros(len(trackers))
                for i in range(len(trackers)):
                    precision, precision_at_level = self.calc_precision(filename, target, trackers[i], thresholds, precision_level, absolute)
                    precisions_at_level[i] = precision_at_level
                inds = np.argsort(precisions_at_level)[::-1]
                for i in inds:
                    precision, precision_at_level = self.calc_precision(filename, target, trackers[i], thresholds, precision_level, absolute)
                    plt.plot(thresholds,
                             precision,
                             label=trackers[i] + (" [%2.3f]" % precision_at_level),
                             color=self._styles[i]['c'],
                             linestyle=self._styles[i]['l'])
                    if filename == "all_files":
                        text = open(self._output_dir + ("/precision_20px_%s.txt" % trackers[i]), "w")
                        text.write("%2.3f\n" % precision_at_level)
                        text.close()
                    plt.legend(loc=4, fontsize='x-small')
                plt.plot([20, 20], [0, 1], '--', color = '0.75')
                ax.set_title("Tracker performance precision plots %s, target %s" % (filename, target))
                ax.set_xlabel("Distance threshold (in pixels)")
                ax.set_ylabel("Fraction of frames")
                if absolute:
                    fig.savefig(self._output_dir + ("/precision_plot_%s_%s_abs.pdf" % (filename, target)))
                else:
                
                    fig.savefig(self._output_dir + ("/precision_plot_%s_%s.pdf" % (filename, target)))

    def calc_precision_rel(self, filename, target, tracker, thresholds, precision_level, absolute):
        data = self._processed_results[filename][target][tracker]['distance_rel']
        data = data[np.where(data >= 0)]
        precision = np.zeros(thresholds.shape)
        for j, x in enumerate(thresholds):
            if absolute:
                precision[j] = 1.0*np.where(data < x)[0].shape[0]/self._processed_results[filename][target]["ground_truth"]['total_true_positive']
            else:
                if self._processed_results[filename][target][tracker]['total_true_positive']>0:
                    precision[j] = 1.0*np.where(data < x)[0].shape[0]/self._processed_results[filename][target][tracker]['total_true_positive']
                else:
                    precision[j] = 0
        return precision, precision[np.argmin(np.abs(thresholds-precision_level))]

    def plot_precision_rel(self, plot_individual_passes=False, precision_level=0.2, absolute=True):
        """
        Plots a modified version of the precision plot, where the x-axis is in relative units instead of pixels

        :return:
        """
        if plot_individual_passes:
            names = self._processed_results.keys()
        else:
            names = ["all_files"]
        thresholds = np.arange(0.0, 1.0, 0.001)
        for filename in names:
            for target in self._processed_results[filename].keys():
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111)
                trackers = sorted(self._processed_results[filename][target].keys())
                precisions_at_level = np.zeros(len(trackers))
                for i in range(len(trackers)):
                    precision, precision_at_level = self.calc_precision_rel(filename, target, trackers[i], thresholds, precision_level, absolute)
                    precisions_at_level[i] = precision_at_level
                inds = np.argsort(precisions_at_level)[::-1]
                for i in inds:
                    precision, precision_at_level = self.calc_precision_rel(filename, target, trackers[i], thresholds, precision_level, absolute)
                    plt.plot(thresholds,
                             precision,
                             label=trackers[i] + (" [%2.3f]" % precision_at_level),
                             color=self._styles[i]['c'],
                             linestyle=self._styles[i]['l'])
                    plt.legend(loc=4, fontsize='x-small')
                    if filename == "all_files":
                        text = open(self._output_dir + ("/precision_rel_%s.txt" % trackers[i]), "w")
                        text.write("%2.3f\n" % precision_at_level)
                        text.close()
                ax.set_title("Tracker performance precision plots %s, target %s" % (filename, target))
                ax.set_xlabel("Distance threshold (relative)")
                ax.set_ylabel("Fraction of frames")
                if absolute:
                    fig.savefig(self._output_dir + ("/precision_rel_plot_%s_%s_abs.pdf" % (filename, target)))
                else:
                
                    fig.savefig(self._output_dir + ("/precision_rel_plot_%s_%s.pdf" % (filename, target)))

    def calc_success(self, filename, target, tracker, absolute):
        data = self._processed_results[filename][target][tracker]['overlap']
        data = data[np.where(data >= 0)]
        steps = 100
        success = np.zeros(steps,)
        xax = np.linspace(0, 1.0, steps)
        area = 0
        for (i, x) in enumerate(xax):
            th = x
            if absolute:
                success[i] = 1.0*np.where(data > th)[0].shape[0]/self._processed_results[filename][target]["ground_truth"]['total_true_positive']
            else:
                if self._processed_results[filename][target][tracker]['total_true_positive']>0:
                    success[i] = 1.0*np.where(data > th)[0].shape[0]/self._processed_results[filename][target][tracker]['total_true_positive']
                else:
                    success[i] = 0

            area += success[i]*1.0/steps
        return success, xax, area

    def plot_success(self, plot_individual_passes=False, absolute=True):
        """
        Plots succes plots as described by [TRBEN13]_.

        :param plot_individual_passes: plot for individual movies, default True
        :return:
        """
        if plot_individual_passes:
            names = self._processed_results.keys()
        else:
            names = ["all_files"]
        for filename in names:
            for target in self._processed_results[filename].keys():
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111)
                trackers = sorted(self._processed_results[filename][target].keys())
                areas = np.zeros(len(trackers))
                for j in range(len(trackers)):
                    success, xax, area = self.calc_success(filename, target, trackers[j], absolute)
                    areas[j] = area
                # sort the trackers by their area score first, so that the legend will appear in rank order
                inds = np.argsort(areas)[::-1]
                for j in inds:
                    success, xax, area = self.calc_success(filename, target, trackers[j], absolute)
                    plt.plot(xax,
                             success,
                             label=trackers[j] + (" [%2.3f]" % area),
                             color=self._styles[j]['c'],
                             linestyle=self._styles[j]['l']
                             )
                    if filename == "all_files":
                        text = open(self._output_dir + ("/success_%s.txt" % trackers[j]), "w")
                        text.write("%2.3f\n" % area)
                        text.close()
                    plt.legend(fontsize='x-small')
                ax.set_title("Tracker performance success plots %s, target %s" % (filename, target))
                ax.set_xlabel("Overlap threshold")
                ax.set_ylabel("Fraction of frames")
                if absolute:
                    fig.savefig(self._output_dir + ("/success_plot_%s_%s_abs.pdf" % (filename, target)))
                else:
                    fig.savefig(self._output_dir + ("/success_plot_%s_%s.pdf" % (filename, target)))

    def plot_performance(self, plot_individual_passes=True):
        """
        Plots the tracker performance (overlap and relative distance to target) together with
        times of target absence and false positives and false negatives against time.

        :param plot_individual_passes: plot for individual movies, default True
        :return:
        """
        names = self._raw_results.keys()
        if plot_individual_passes:
            names = self._processed_results.keys()
        else:
            names = ["all_files"]
        for filename in names:
            for target in self._processed_results[filename].keys():
                pdf = PdfPages(self._output_dir + ("/performance_plot_%s_%s.pdf" % (filename, target)))
                for tracker in self._processed_results[filename][target].keys():
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111)
                    overlap = self._processed_results[filename][target][tracker]['overlap']
                    presence = self._processed_results[filename][target][tracker]['presence']
                    distance = self._processed_results[filename][target][tracker]['distance_rel']
                    xax = np.arange(0, overlap.shape[0], 1)
                    plt.plot(xax, overlap, color='k', label="Overlap")
                    plt.plot(xax, distance, color='k', linestyle=':', label="Distance")
                    false_positive = np.zeros_like(presence)
                    false_positive[np.where(presence == 1)] = np.sqrt(2)
                    false_negative = np.zeros_like(presence)
                    false_negative[np.where(presence == 2)] = np.sqrt(2)
                    true_negative = np.zeros_like(presence)
                    true_negative[np.where(presence == 3)] = np.sqrt(2)
                    plt.plot(xax, false_positive, 'r', linewidth=0.1, zorder=1, label='False positive')
                    plt.fill_between(xax, false_positive, 0, color='r', alpha=0.3)
                    plt.plot(xax, false_negative, 'm', linewidth=0.1, zorder=1, label='False negative')
                    plt.fill_between(xax, false_negative, 0, color='m', alpha=0.3)
                    plt.plot(xax, true_negative, 'g', linewidth=0.1, zorder=1, label='True negative')
                    plt.fill_between(xax, true_negative, 0, color='g', alpha=0.3)
                    plt.legend(fontsize='x-small')
                    ax.set_title("Tracker %s on %s, target %s" % (tracker, filename, target))
                    ax.set_xlabel("Time (frames)")
                    ax.set_ylabel("Overlap/Distance")
                    plt.close()
                    pdf.savefig(fig)
                pdf.close()

    def plot_presence(self, plot_individual_passes=True):
        """
        This method plots the presence pie chart, that is the pie chart containing the number of frames being
        true positives, true negatives, false positives and false negatives.

        Having good majority of true positives and true negatives against false positives and false negatives is a
        necessary but not sufficient condition of good tracking.

        :param plot_individual_passes: plot for individual movies, default True
        :return:
        """
        names = self._raw_results.keys()
        if plot_individual_passes:
            names = self._processed_results.keys()
        else:
            names = ["all_files"]
        for filename in names:
            for target in self._processed_results[filename].keys():
                pdf = PdfPages(self._output_dir + ("/presence_plot_%s_%s.pdf" % (filename, target)))
                for tracker in self._processed_results[filename][target].keys():
                    fig = plt.figure(figsize=(7, 6))
                    ax = fig.add_subplot(111)
                    labels = ["True positive", "True negative", "False positive", "False negative"]
                    sizes = [self._processed_results[filename][target][tracker]['true_positive'],
                             self._processed_results[filename][target][tracker]['true_negative'],
                             self._processed_results[filename][target][tracker]['false_positive'],
                             self._processed_results[filename][target][tracker]['false_negative']]
                    colors = ['yellowgreen', 'lightskyblue', 'red', 'magenta']
                    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True)
                    plt.axis('equal')
                    ax.set_title("Tracker %s on %s, target %s" % (tracker, filename, target))
                    plt.close()
                    pdf.savefig(fig)
                pdf.close()

    def plot_timing(self, plot_individual_passes=True):
        """
        This method plots the timing plot for individual movie runs and combined over the entire dataset.

        :param plot_individual_passes: plot for individual movies, default True
        :return:
        """
        names = self._timing_results.keys()
        combined_norm = {}
        for filename in names:
            for target in self._processed_results[filename].keys():
                for tracker in self._processed_results[filename][target].keys():
                    if tracker == "ground_truth" or tracker.startswith("GT_pert"):
                        continue
                    timing = np.array(self._timing_results[filename][target][tracker]['norm'])
                    if tracker not in combined_norm.keys():
                        combined_norm[tracker] = {"norm": timing}
                    else:
                        combined_norm[tracker]['norm'] = np.append(combined_norm[tracker]['norm'], timing)
        self._timing_results['all_files'] = {}
        self._timing_results['all_files']['all_targets'] = combined_norm
        names = self._timing_results.keys()
        for filename in names:
            for target in self._processed_results[filename].keys():
                pdf = PdfPages(self._output_dir + ("/timing_plot_%s_%s.pdf" % (filename, target)))
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111)
                for (j, tracker) in enumerate(sorted(self._processed_results[filename][target].keys())):
                    if tracker == "ground_truth" or tracker.startswith("GT_pert"):
                        continue
                    timing = np.array(self._timing_results[filename][target][tracker]['norm'])
                    xax = np.arange(0, timing.shape[0], 1)
                    plt.plot(xax,
                             1000*timing,
                             label=tracker,
                             color=self._styles[j]['c'],
                             linestyle=self._styles[j]['l'])
                plt.legend(fontsize='x-small')
                plt.ylabel("Execution time (ms)")
                plt.xlabel("Time (frames)")
                plt.yscale('log')
                ax.set_title("Timing per pixel on %s, target %s" % (filename, target))
                plt.close()
                pdf.savefig(fig)
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111)
                data = []
                labels = []
                for tracker in self._processed_results[filename][target].keys():
                    if tracker == "ground_truth" or tracker.startswith("GT_pert"):
                        continue
                    timing = np.array(self._timing_results[filename][target][tracker]['norm'])
                    data.append(1000 * timing)
                    labels.append(tracker)
                plt.boxplot(data, labels=labels, showmeans=True)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=15)
                plt.title("Tracker timing")
                try:
                    plt.yscale('log')
                except:
                    plt.yscale('linear')  # some times log axis fails
                plt.ylabel("Execution time (ms)")
                pdf.savefig(fig)
                pdf.close()

    def calc_accuracy(self, filename, target, tracker, thresholds):
        data = np.abs(self._processed_results[filename][target][tracker]['bounding_box_dists_rel'])*2  # *2 so that 1.0 means edge of bounding box instead of 0.5
        correct = np.zeros(thresholds.shape)
        for j, x in enumerate(thresholds):
            correct[j] = np.mean(np.bitwise_and(data[:, 0]<=x, data[:, 1]<=x))
        return correct, correct[np.argmin(np.abs(thresholds-1.0))]  # accuracy is measured at 1.0

    def plot_accuracy(self, plot_individual_passes=False):
        """
        Plots the accuracy of tracking as a function of distance away from center in units of bounding box size.
        If the center of the tracking bounding box lies within x*(ground truth bounding box) or the tracker correctly reports
        the absense of the target (regardless of x) the analysis reports the frame as correctly tracked, where x is varied between 0 and 10.
        When x==1, the ground truth bounding box is used.  Note the area of the scaled bounding box grows as x^2.
        """
        if plot_individual_passes:
            names = self._processed_results.keys()
        else:
            names = ["all_files"]
        thresholds = np.arange(0.0, 10.0, 0.01)
        for filename in names:
            for target in self._processed_results[filename].keys():
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111)
                trackers = sorted(self._processed_results[filename][target].keys())
                accuracies = np.zeros(len(trackers))
                for i in range(len(trackers)):
                    correct, accuracy = self.calc_accuracy(filename, target, trackers[i], thresholds)
                    accuracies[i] = accuracy
                inds = np.argsort(accuracies)[::-1]
                for i in inds:
                    correct, accuracy = self.calc_accuracy(filename, target, trackers[i], thresholds)
                    plt.plot(thresholds,
                             correct,
                             label=trackers[i] + (" [%2.3f]" % accuracy),
                             color=self._styles[i]['c'],
                             linestyle=self._styles[i]['l'])
                    if filename == "all_files":
                        text = open(self._output_dir + ("/accuracy_%s.txt" % trackers[i]), "w")
                        text.write("%2.3f\n" % accuracy)
                        text.close()
                    plt.legend(loc=4, fontsize='x-small')
                plt.plot([1, 1], [0, 1], '--', color = '0.75')
                ax.set_title("Tracker accuracy plots %s, target %s" % (filename, target))
                ax.set_xlabel("bounding box linear size (1 means within ground truth bounding box)")
                ax.set_ylabel("Fraction of frames tracked correctly")
                fig.savefig(self._output_dir + ("/accuracy10_plot_%s_%s.pdf" % (filename, target)))
                ax.set_xlim(0, 2)
                fig.savefig(self._output_dir + ("/accuracy_plot_%s_%s.pdf" % (filename, target)))
                ax.set_xlim(0.03, 10)
                plt.xscale('log')
                fig.savefig(self._output_dir + ("/accuracy_log_plot_%s_%s.pdf" % (filename, target)))

    def print_results(self, file=sys.stdout, moviename=None):
        """
        Prints the most basic processed results of tracker benchmark

        :param results: processed results as returned by process results
        :return:
        """
        if moviename is None:
            flist = self._processed_results.keys()
        else:
            flist = [moviename]
        for moviename in flist:
            for target in self._processed_results[moviename].keys():
                print >>file, "File " + moviename
                print >>file, "Target " + target
                for tracker in self._processed_results[moviename][target].keys():
                    print >>file, "Tracker " + tracker
                    print >>file, "Total frames         %d" % self._processed_results[moviename][target][tracker]['total_frames']
                    print >>file, "True positive        %2.2f %%" % self._processed_results[moviename][target][tracker]['true_positive']
                    print >>file, "True negative        %2.2f %%" % self._processed_results[moviename][target][tracker]['true_negative']
                    print >>file, "False positive       %2.2f %%" % self._processed_results[moviename][target][tracker]['false_positive']
                    print >>file, "False negative       %2.2f %%" % self._processed_results[moviename][target][tracker]['false_negative']
                    print >>file, "Presence score       %2.2f %%" % self._processed_results[moviename][target][tracker]['presence_score']
                    print >>file, "Average distance     %2.2f pixels" % self._processed_results[moviename][target][tracker]['avg_distance_pix']
                    print >>file, "Average distance rel %0.4f" % self._processed_results[moviename][target][tracker]['avg_distance_rel']
                    data = np.abs(self._processed_results[moviename][target][tracker]['bounding_box_dists_rel'])*2  # *2 so that 1.0 means edge of bounding box instead of 0.5
                    print >>file, "Correctly tracked    %2.2f %%" % (100.0 * np.mean(np.bitwise_and(data[:, 0] <= 1.0, data[:, 1] <= 1.0)))

    def get_summary(self):
        self.print_results(moviename="all_files")

    def run(self, file_list, tracker_list, print_on_the_fly=True, channel="default", remove_downloads=True):
        """
        Runs the evaluation on a list of files. Appends result dicts in a list

        :param file_list: list of files to be processed
        :param tracker_list: list of tracker objects to be evaluated
        :param print_on_the_fly: bool flag that will trigger instant performance message after each movie
        :return: list of result dicts
        """
        summary = open(self._output_dir+"/summary.txt", "aw", 1)

        for (i, file) in enumerate(file_list):
            if file.endswith(".pkl"):
                print "Processing " + file + " %d of %d" % (i + 1, len(file_list))
                local_path = self.storage.get(path=file)
                new_results = self.evaluate_on_file(local_path, tracker_list, channel=channel)
                if remove_downloads:
                    try:
                        os.remove(local_path)
                    except OSError:
                        print "Could not remove %s" % local_path
                if new_results:
                    self.process_results(os.path.basename(local_path))
                    if print_on_the_fly:
                        self.print_results(moviename=os.path.basename(local_path))
                        self.print_results(file=summary, moviename=os.path.basename(local_path))
                    self.save_results()  # store the intermediate results
        for tracker in tracker_list:
            tracker.finish()
        self.print_results(file=summary, moviename="all_files")
        summary.close()

    def plot_all(self):
        self.plot_accuracy()
        self.plot_precision()
        self.plot_precision_rel()
        self.plot_success()
        self.plot_performance()
        self.plot_presence()
        self.plot_timing()


def time_dir_name(prefix, suffix):
    lt = time.localtime()
    return prefix + ("_%d_%d_%d__%d_%d_" % (lt[0], lt[1], lt[2], lt[3], lt[4])) + suffix


if __name__ == "__main__":
    have_display = True
    try:
        from Tkinter import Tk
        Tk()
    except:
        have_display = False
        print "No display detected, turning off visualizations."

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--channel", type=str, default="default", help="Channel")
    parser.add_argument("-r", "--run", help="Run the benchmark (otherwise will just process data)", action="store_true")
    parser.add_argument("-o", "--output", type=str, default=time_dir_name("~/benchmark_results/run", ""), help="Results file to either store or load from")
    parser.add_argument("-disp", "--display", type=int, default=True, help="set to False to force no display")
    parser.add_argument("-m", "--sparse_manager_file", type=str, default="", help="SparseManager saved file to load")
    parser.add_argument("-s", "--set", help="Name of the dataset", type=str, default="green_ball")
    parser.add_argument("-res", "--resolution", help="resolution to force all videos to, eg. '96,96' or 'full', default 'full'", type=str, default="full")
    parser.add_argument("-rd", "--remove_downloads", help="removes downloads after use, default False", type=int, default=False)
    args = parser.parse_args()
    ts = PVM_Storage.Storage()
    res = None
    if args.resolution.lower() != "full":
        res = tuple(np.fromstring(args.resolution, dtype=int, sep=','))
        
    if not args.display:
        have_display = False

    TB = TrackerBenchmark(output_dir=args.output, resolution=res, have_display=have_display, storage=ts)
    if args.run:
        files = []
        for (file, channel) in tracker_datasets.sets["%s_testing" % args.set]:
            files.append(file)
        trackers = []
        TB.run(files, trackers, channel=args.channel, remove_downloads=args.remove_downloads)
        TB.save_results()
    TB.get_summary()
    TB.plot_all()
