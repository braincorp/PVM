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


class NullVisionTracker(GenericVisionTracker):
    """
    This class exposes the null vision tracker which just always
    returns its priming bounding box

    """
    def __init__(self, scaling=1.0, new_name=None):
        """
        Initialize the tracker
        """
        if new_name is None:
            self.name = 'null_tracker'
        else:
            self.name = new_name
        self.scaling = scaling

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
        self.bbox = bounding_region.copy()
        self.bbox.scale(self.scaling)
        if not self._primed:
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
        self._primed = False
        return self.bbox.copy()

    def get_heatmap(self, heatmap_name=None):
        return self.bbox.get_mask()
