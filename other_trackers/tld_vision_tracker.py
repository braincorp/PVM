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
import other_trackers.tld as tld
import cv2
from PVM_tools.bounding_region import BoundingRegion


class TLDVisionTracker(GenericVisionTracker):
    """
    This class exposes the open tld tracker implemented in C++ by http://www.gnebehay.com/tld/.
    Originally OpenTLD that was originally published in MATLAB by Zdenek Kalal.

    """
    def __init__(self):
        """
        Initialize the tracker
        """
        self.name = 'tld_tracker'
        self.reset()

    def reset(self):
        """
        Reset the tracker
        :return:
        """
        self.tld = None
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

        if not self._primed:
            self.tld = tld.TLD2()
            self._primed = True

        self.height, self.width = im.shape[:2]
        self.tld.set_width_and_height((self.width, self.height))

        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        img_cvmat = cv2.cv.fromarray(im)
        if bounding_box[0] + bounding_box[2] > self.width:
            bounding_box[2] = self.width - bounding_box[0]
        if bounding_box[1] + bounding_box[3] > self.height:
            bounding_box[3] = self.height - bounding_box[1]

        self.tld.selectObject(img_cvmat, tuple([int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])]))

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
        img_cvmat = cv2.cv.fromarray(im)
        self.tld.processImage(img_cvmat)
        self.bounding_box = self.tld.getCurrBB()
        self.confidence = self.tld.currConf
        if self.bounding_box == ():
            self.bbox = BoundingRegion()
        else:
            self.bbox = BoundingRegion(image_shape=(im.shape[0], im.shape[1], 3), box=self.bounding_box, confidence=self.confidence)
        return self.bbox.copy()

    def get_heatmap(self, heatmap_name=None):
        return self.bbox.get_mask()
