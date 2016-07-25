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
import PVM_framework.CoreUtils as CoreUtils
import PVM_framework.PVM_Create as PVM_Create
import PVM_framework.AbstractExecutionManager as AbstractExecutionManager
import logging
import cv2
from PVM_tools.abstract_tracker import GenericVisionTracker
from PVM_tools.bounding_region import BoundingRegion
import numpy as np


class Manager(AbstractExecutionManager.ExecutionManager):
    def __init__(self, prop_dict, steps_to_run):
        self.prop_dict = prop_dict
        self.steps_to_run = steps_to_run
        self._running = True

    def start(self):
        """
        This will be called right before the simulation starts
        """
        pass

    def fast_action(self):
        """
        This is the time between steps of execution
        Data is consistent, but keep this piece absolutely minimal
        """
        pass

    def slow_action(self):
        """
        This is while the workers are running. You may do a lot of work here
        (preferably not more than the time of execution of workers).
        """
        pass

    def running(self):
        """
        While returning True the simulation will keep going.
        """
        return self._running

    def finish(self):
        pass


class PVMVisionTracker(GenericVisionTracker):
    """
    This class exposes the null vision tracker which just always
    returns its priming bounding box

    """
    def __init__(self, filename="", remote_filename="", cores="4", storage=None, steps_per_frame=1):
        """
        Initialize the tracker
        """
        self.name = 'PVMtracker'
        if filename == "":
            filename = storage.get(remote_filename)
        self.prop_dict = CoreUtils.load_model(filename)
        logging.info("Loaded the dictionary %s", filename)
        PVM_Create.upgrade_dictionary_to_ver1_0(self.prop_dict)
        self.prop_dict['num_proc'] = int(cores)
        for k in range(len(self.prop_dict['learning_rates'])):
            self.prop_dict['learning_rates'][k][0] = 0.0
            logging.info("Setting learning rate in layer %d to zero" % k)
        self.prop_dict["readout_learning_rate"][0] = 0.0
        logging.info("Setting readout learning rate to zero")
        self.manager = Manager(self.prop_dict, 1000)
        self.executor = CoreUtils.ModelExecution(prop_dict=self.prop_dict, manager=self.manager, port=9100)
        self.executor.start(blocking=False)
        self.threshold = 32
        self.image_size = self.prop_dict['input_array'].shape[:2][::-1]
        self.readout_heatmap = np.zeros(self.image_size, dtype=np.float)
        self.step_per_frame = steps_per_frame

    def reset(self):
        """
        Reset the tracker
        :return:
        """
        self._primed = False
        self.bbox = None

    def _prime(self, im, bounding_region):
        """
        prime tracker on image and bounding box

        :param im: input image (3 - channel numpy array)
        :type im: numpy.ndarray
        :param bounding_region: initial bounding region of the tracked object
        :type bounding_region: PVM_tools.bounding_region.BoundingRegion
        """
        if not self._primed:
            logging.info("Priming the PVM tracker")
            logging.info("Setting up the tracker threshold to %d" % self.threshold)
            self._primed = True

    def _track(self, im):
        """
        Track on given image, rseturn a bounding box

        :param im: image (3 - channel numpy array)
        :type im: numpy.ndarray
        :return: bounding box of the tracker object
        :rtype: PVM_tools.bounding_region.BoundingRegion
        """
        #  this is required so that tracker is re-initialized if it is primed again
        current_frame = cv2.resize(im, dsize=self.image_size)
        self.prop_dict['input_array'][:] = current_frame
        self.prop_dict['input_array_float'][:] = current_frame.astype(np.float)/255
        for i in range(self.step_per_frame):
            self.executor.step()
        norm = 1.0 / len(self.prop_dict['predicted_readout_arrays'])
        self.readout_heatmap[:] = 0
        for k in self.prop_dict['predicted_readout_arrays']:
            self.readout_heatmap[:] += cv2.resize(k.view(np.ndarray), dsize=self.image_size) * norm
        am = np.unravel_index(np.argmax(self.readout_heatmap), self.readout_heatmap.shape)
        if len(am) == 3:
            for d in range(3):
                if am[2] != d:
                    self.readout_heatmap[:, :, d] = 0

        self.heatmap = self.readout_heatmap.view(np.ndarray)
        if len(self.heatmap.shape) == 3:
            self.heatmap = np.max(self.heatmap, axis=2)
        self.heatmap = cv2.resize(self.heatmap, dsize=(im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        self.heatmap = (self.heatmap * 255).astype(np.uint8)
        if np.max(self.heatmap) > (np.median(self.heatmap)+self.threshold):
            threshold=(np.max(self.heatmap)-np.median(self.heatmap))*0.5+np.median(self.heatmap)
            ret, thresh = cv2.threshold(self.heatmap, threshold, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                max_cnt=None
                for cnt in contours:
                    pt = np.unravel_index(np.argmax(self.heatmap), self.heatmap.shape)
                    if cv2.pointPolygonTest(cnt, (pt[1], pt[0]), False)>=0:
                        max_cnt = cnt
                        break
                if max_cnt is None:
                    self.bbox = BoundingRegion()
                else:
                    x, y, w, h = cv2.boundingRect(max_cnt)
                    if w*h > 25:
                        self.bbox = BoundingRegion(image_shape=im.shape, box=[x, y, w, h])
                        self.bbox.scale(1.1)
                    else:
                        self.bbox = BoundingRegion()
            else:
                self.bbox = BoundingRegion()
        else:
            self.bbox = BoundingRegion()
        self._primed = False
        return self.bbox.copy()

    def get_heatmap(self, heatmap_name=None):
        return self.heatmap()

    def finish(self):
        self.manager._running = False
        self.executor.step()
        self.executor.finish()
