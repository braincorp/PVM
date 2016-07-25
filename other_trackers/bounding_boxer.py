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
import numpy as np
import cv2
from PVM_tools.abstract_bounding_boxer import AbstractBoundingBoxer
from PVM_tools.bounding_region import BoundingRegion


class CamshiftBoundingBoxer(AbstractBoundingBoxer):
    """
    Given a prob-map of pixel membership, return a bounding box: (PVM_tools.bounding_region.BoundingRegion)

    This class uses the opencv CamShift method to estimate the new position of the bounding box
    based on the previous one. In the case the previous bounding box is empty, then a search over the
    entire map is performed to estimate the new position.
    """

    def __init__(self, recovery_kernel_size=20, min_recovered_box_area=400, threshold_retrieve=0.2):
        """
        :param recovery_kernel_size: when searching for a new box, the probablity map will be filtered with a box filter, to smooth out single peaks and find a better candidate. This is the size of the box kernal.
        :type recovery_kernel_size: int
        :param min_recovered_box_area: after finding a box, assert that width x height is at least this number of pixels. Otherwise do not accept that as a good box and return an empty box.
        :type min_recovered_box_area: int
        :param threshold_retrieve: When object the is lost, that is the minimal confidence the tracker must get to consider that a new candidate is considered as the new target Lower value (ex 0.1) provides better recovery but more false positive. Typical value are between 0.1 and 0.2
        :type threshold_retrieve: float
        """
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self._last_bb = None
        self.confidence = 1.
        self.recovery_kernel_size = recovery_kernel_size
        self.min_recovered_box_area = min_recovered_box_area
        self.threshold_retrieve = threshold_retrieve

    def reset(self):
        """
        Reset the object.
        """
        self._last_bb = None

    def _find_new_box(self, heatmap):
        heatmap = cv2.boxFilter(heatmap, ddepth=-1, ksize=(self.recovery_kernel_size, self.recovery_kernel_size), borderType=cv2.BORDER_REPLICATE)
        # new_bb = np.array(np.unravel_index(np.argmax(conv_prob), conv_prob.shape))[::-1]
        # new_bb = np.array(tuple(new_bb - max(1, min(10, self.recovery_kernel_size/2))) + (min(20, self.recovery_kernel_size), ) * 2)
        # _, bb_reset = cv2.CamShift(heatmap, tuple(new_bb), self.term_crit)
        peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        _, heatmap = cv2.threshold(heatmap, np.max(heatmap)*0.9, 255, cv2.THRESH_BINARY)
        _, bb_reset = cv2.floodFill(heatmap, None, tuple(peak[::-1]), 255, loDiff=10, flags=cv2.FLOODFILL_FIXED_RANGE)
        bb_area = bb_reset[2] * bb_reset[3]
        mean_prob_reset = np.sum(heatmap[bb_reset[1]:bb_reset[1] + bb_reset[3], bb_reset[0]:bb_reset[0] + bb_reset[2]]) / float(bb_area)
        return bb_reset, bb_area, mean_prob_reset

    def process(self, heatmap):
        """
        The main processing method. Given the probability map, returns a bounding region

        :param heatmap: the map (e.g. result of histogram backprojection etc.). Its supposed to be a numpy array of float.
        :type heatmap: numpy.ndarray
        :return: bounding_box
        :rtype: PVM_tools.bounding_region.BoundingRegion
        """
        self._last_prob = heatmap
        harea = np.prod(heatmap.shape)

        if self._last_bb is None or self._last_bb.empty:
            (bb, area, mean_prob) = self._find_new_box(heatmap)
            confidence = mean_prob
            if bb[2] > 0 and bb[3] > 0 and np.prod(bb[2:])<2*harea/3:
                self._last_bb = BoundingRegion(image_shape=(heatmap.shape[0], heatmap.shape[1], 3),
                                               box=np.array(bb, dtype=np.int),
                                               confidence=confidence)
            else:
                # Empty bounding box
                self._last_bb = BoundingRegion()
        else:
            # Step 1 go with the last bbox
            _, bb0 = cv2.meanShift(heatmap, tuple(self._last_bb.get_box_pixels()), self.term_crit)
            area0 = np.prod(bb0[2:])
            mean_prob0 = np.sum(heatmap[bb0[1]:bb0[1] + bb0[3], bb0[0]:bb0[0] + bb0[2]]) / (1e-12 + area0)
            _, bb = cv2.CamShift(heatmap, tuple(self._last_bb.get_box_pixels()), self.term_crit)
            area = np.prod(bb[2:])
            mean_prob = np.sum(heatmap[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]) / (1e-12 + area)
            if mean_prob > self.threshold_retrieve and area > self.min_recovered_box_area and area < 2*harea/3:
                self._last_bb = BoundingRegion(image_shape=(heatmap.shape[0], heatmap.shape[1], 3),
                                               box=np.array(bb, dtype=np.int),
                                               confidence=mean_prob)
            elif mean_prob0 > self.threshold_retrieve and area0 > self.min_recovered_box_area:
                # Go with the mean shift box, changing size apparently does no good.
                return self._last_bb
            else:
                # Empty bounding box
                self._last_bb = BoundingRegion()

        return self._last_bb

    def set_current_bounding_box(self, bounding_box):
        """
        If the bounding box is somehow available (e.g. in priming) that values can be passed in here to affect
        future computation (new boxes will be found in reference to this one)

        :param bounding_box: A bounding region object
        :type bounding_box: PVM_tools.bounding_region.BoundingRegion
        """
        self._last_bb = bounding_box


class FloodFillBoundingBoxer(AbstractBoundingBoxer):
    """
    Given a prob-map of pixel membership, return a bounding box: (PVM_tools.bounding_region.BoundingRegion)

    This class uses the opencv FloodFill from the peak of the heatmap to estimate the new position of the bounding box.
    """
    def __init__(self, heatmap_threshold=200, distance_threshold=20):
        self.heatmap_threshold = heatmap_threshold
        self.distance_threshold = distance_threshold
        self.recovery_kernel_size = 20

    def reset(self):
        pass

    def process(self, heatmap, previous_bb=None, peak=None):
        """
        This method computes a bounding box around the heatmap peak.
        Arguments:
        heatmap - target position heat map (may have multiple local maxima).
        peak - [y, x] of the most likely target position (usually the highest peak of the heatmap).
               This argument tells compute_bounding_box() which local maximum to choose.
        returns a bounding box array (x_upper_left,y_upper_left,width,height) or None if it can't be found
        """

        if np.max(heatmap) == 0.0:
            return BoundingRegion()
        heatmap = cv2.boxFilter(heatmap, ddepth=-1, ksize=(self.recovery_kernel_size, self.recovery_kernel_size), borderType=cv2.BORDER_REPLICATE)

        if peak is None:
            peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        if np.issubdtype(heatmap.dtype, np.float):
            _, heatmap = cv2.threshold(heatmap, self.heatmap_threshold*(1.0/255), 255, cv2.THRESH_BINARY)
        else:
            _, heatmap = cv2.threshold(heatmap, self.heatmap_threshold, 255, cv2.THRESH_BINARY)

        if heatmap[peak[0], peak[1]] != 255:
            cors = np.nonzero(heatmap)
            new_peak = None
            if len(cors[0]) > 0:
                dist2 = (cors[0] - peak[0])**2 + (cors[1] - peak[1])**2
                ind=np.argmin(dist2)
                if dist2[ind] < self.distance_threshold**2:
                    new_peak = np.array([cors[0][ind], cors[1][ind]])
            if new_peak is None:
                return BoundingRegion(image_shape=(heatmap.shape[0], heatmap.shape[1], 3), box=np.array([peak[1], peak[0], 1, 1]))
            peak = new_peak

        _, bounding_box = cv2.floodFill(heatmap, None, tuple(peak[::-1]), 255, loDiff=10, flags=cv2.FLOODFILL_FIXED_RANGE)

        return BoundingRegion(image_shape=(heatmap.shape[0], heatmap.shape[1], 3), box=np.asarray(bounding_box))

    def set_current_bounding_box(self, bounding_region):
        pass
