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

from PVM_tools.abstract_tracker import GenericVisionTracker
import other_trackers.struck_bindings as struck
import cv2
from PVM_tools.bounding_region import BoundingRegion

struck_config = """
# quiet mode disables all visual output (for experiments).
quietMode = 0

# debug mode enables additional drawing and visualization.
debugMode = 1

# base path for video sequences.
sequenceBasePath = sequences

# path for output results file.
# comment this out to disable output.
#resultsPath = log.txt

# video sequence to run the tracker on.
# comment this out to use webcam.
sequenceName = girl

# frame size for use during tracking.
# the input image will be scaled to this size.
frameWidth = 320
frameHeight = 240

# seed for random number generator.
seed = 0

# tracker search radius in pixels.
searchRadius = 30

# SVM regularization parameter.
svmC = 100.0
# SVM budget size (0 = no budget).
svmBudgetSize = 100

# image features to use.
# format is: feature kernel [kernel-params]
# where:
#   feature = haar/raw/histogram
#   kernel = gaussian/linear/intersection/chi2
#   for kernel=gaussian, kernel-params is sigma
# multiple features can be specified and will be combined
feature = haar gaussian 0.2
#feature = raw gaussian 0.1
#feature = histogram intersection
"""


class StruckTracker(GenericVisionTracker):
    """
    TODO
    """
    def __init__(self):
        """
        Initialize the tracker
        """
        self.name = 'struck_tracker'
        f = open("config.txt", "w")
        f.write(struck_config)
        f.close()
        self.reset()

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
        self.bbox=bounding_region
        bounding_box = bounding_region.get_box_pixels()
        struck.STRUCK_init(im, bounding_box)
    
    def _track(self, im):
        """
        Track on given image, return a bounding box

        :param im: image (3 - channel numpy array)
        :type im: numpy.ndarray
        :return: bounding box of the tracker object
        :rtype: PVM_tools.bounding_region.BoundingRegion
        """
        #  this is required so that tracker is re-initialized if it is primed again
        self._primed = False
        struck.STRUCK_track(im)
        struck_bbox = struck.STRUCK_get_bbox()
        self.bounding_box = [struck_bbox["xmin"], struck_bbox["ymin"], struck_bbox["width"], struck_bbox["height"]]
        if self.bounding_box == ():
            self.bbox = BoundingRegion()
        else:
            self.bbox = BoundingRegion(image_shape=(im.shape[0], im.shape[1], 3), box=self.bounding_box)
        return self.bbox.copy()

    def get_heatmap(self, heatmap_name=None):
        return self.bbox.get_mask()
