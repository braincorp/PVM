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
import cv2
from PVM_tools.abstract_tracker import GenericVisionTracker
from backprojection import ColorHistogramBackProjection
from other_trackers.bounding_boxer import CamshiftBoundingBoxer
from PVM_tools.bounding_region import BoundingRegion


class BasicHistogramBackprojectionTracker(GenericVisionTracker):
    """
    Basic color histogram backprojection tracker as described in [DB90]_. The idea can be summarized as follows:

        1. Calculate a color histogram of the image using all or a subset of channels

        2. Calculate a color histogram of the region of interest (ROI)

        3. Divide the ROI histogram by the whole image histogram (normalization)

        4. Backproject the histogram to the image (each pixel of the new heatmap containst the value of the
            histogram bucket to which the original pixel in the image belonged)

        5. Estimate the new position of the target. In this implementation a simple kinematic tracker based
            on the camshift algorithm [B98]_ is used.

    This object is a very general tracker that will likely perform very poorly, but it can be tuned by adjusting
    parameters and selecting appropriate image channels for calculating histograms. In certain cases the histograms
    can be computed on a highly processed feature representations of the image (e.g. orientations or so on).

    This module contains a two related trackers that use this basic tracker as a building block.


    .. [DB90] Swain, M.J.; Ballard, D.H., "Indexing via color histograms," Computer Vision, 1990. Proceedings,
            Third International Conference on , vol., no., pp.390,393, 4-7 Dec 1990
            doi: 10.1109/ICCV.1990.139558

    .. [B98] Bradski, G.R., "Real time face and object tracking as a component of a perceptual user interface,"
            Applications of Computer Vision, 1998. WACV '98. Proceedings., Fourth IEEE Workshop on , vol.,
            no., pp.214,219, 19-21 Oct 1998

    :param bins: shape of the histogram (number of bins in each channel), default (4, 4, 4)
    :type bins: tuple
    :param channels: selection of image channels to be used in histogram calculation, default (0, 1, 2)
    :type channels: tuple
    :param ranges: a set of ranges in each channels to which the histogram should be restricted, default (0, 256, 0, 256, 0, 256)
    :type ranges: tuple
    :param recovery_kernel_size: refer to :class:`other_trackers.bounding_boxer.CamshiftBoundingBoxer`
    :type recovery_kernel_size: int
    :param min_recovered_box_area: refer to :class:`other_trackers.bounding_boxer.CamshiftBoundingBoxer`
    :type min_recovered_box_area: int
    :param confidence_threshold_retrieve: refer to :class:`other_trackers.bounding_boxer.CamshiftBoundingBoxer`
    :type confidence_threshold_retrieve: int
    :param  priming_bbox_rescaling: the factor by which the initial (priming) bounding box should be rescaled before actually priming
    :type priming_bbox_rescaling: float
    :param confidence_threshold: confidence level above which a non empty bounding box will be returned
    :type confidence_threshold: float
    """
    def __init__(self,
                 bins=(4, 4, 4),
                 channels=(0, 1, 2),
                 ranges=(0, 256, 0, 256, 0, 256),
                 recovery_kernel_size=20,
                 min_recovered_box_area=100,
                 confidence_threshold_retrieve=0.2,
                 priming_bbox_rescaling=0.9,
                 confidence_threshold=0.4):
        self._back_projector = ColorHistogramBackProjection(n_bins=bins, channels=channels,
                                                            ranges=ranges,
                                                            use_background=True)
        self._bounding_boxer = CamshiftBoundingBoxer(recovery_kernel_size=recovery_kernel_size,
                                                     min_recovered_box_area=min_recovered_box_area,
                                                     threshold_retrieve=confidence_threshold_retrieve)
        self._priming_bbox_rescaling = priming_bbox_rescaling
        self._confidence_threshold = confidence_threshold
        self._primed = False
        self.name = "Basic Histogram Backprojection Tracker"

    def _prime(self, im, bounding_box=None):
        """
        Prime the tracker

        :param im: priming image
        :type im: numpy.ndarray
        :param bounding_box: initial target bounding box
        :type bounding_box: PVM_tools.bounding_region.BoundingRegion
        :return: True if primed
        :rtype: bool
        """
        if bounding_box is None:
            return
        if self._priming_bbox_rescaling is not None:
            bounding_box.scale(self._priming_bbox_rescaling)
        (x, y, w, h) = bounding_box.get_box_pixels()
        self._back_projector.calculateHistogram(im, (x, y, w, h))
        self.prob = self._back_projector.calculate(im)
        self._bounding_boxer.set_current_bounding_box(bounding_box)
        self._primed = True
        return self._primed

    def _track(self, im):
        """
        Track

        :param im:  image
        :type im: numpy.ndarray
        :return: bounding region
        :rtype: PVM_tools.bounding_region.BoundingRegion
        """
        if not self._primed:
            return BoundingRegion()
        self.prob = self._back_projector.calculate(im)
        bounding_box = self._bounding_boxer.process(heatmap=self.prob)
        if bounding_box.confidence > self._confidence_threshold:
            return bounding_box.copy()
        else:
            return BoundingRegion()

    def reset(self):
        """
        Reset the tracker

        :return:
        """
        self._back_projector.reset()
        self._primed = False

    def get_heatmap(self, heatmap_name=None):
        """
        Get the original histogram backprojection heatmap

        :param heatmap_name: optional argument for tracker which may have multipe heatmaps
        :return: heatmap
        :rtype: numpy.ndarray
        """
        return self.prob


class HSHistogramBackprojectionTracker(GenericVisionTracker):
    """
    This class is a variation of the histogram backprojection tracker for the hue and saturation
    channels. The tracker takes an RGB image, converts it into a proper format and runs through
    a appropriately configured instance of the backprojection tracker.

    :param bins: histogram bins shape (8, 8)
    :type bins: tuple
    :param channels: selection of channels (0, 1)
    :type channels: tuple
    :param ranges: histogram ranges (0, 180, 0, 255)
    :type ranges: tuple
    """

    def __init__(self):
        self.tracker = BasicHistogramBackprojectionTracker(bins=(8, 8), channels=(0, 1), ranges=(0, 180, 0, 255))
        self.name = "HSTracker"

    def _preprocess(self, image):
        """
        Preprocess the image

        :param image:
        :return:
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def _prime(self, im, bounding_box=None, **kwargs):
        """
        Prime the tracker

        :param im: priming image
        :type im: numpy.ndarray
        :param bounding_box: initial target bounding box
        :type bounding_box: PVM_tools.bounding_region.BoundingRegion
        :return: True if primed
        :rtype: bool
        """
        im = self._preprocess(im)
        return self.tracker.prime(im, bounding_box=bounding_box)

    def _track(self, im):
        """
        Track

        :param im:  image
        :type im: numpy.ndarray
        :return: bounding region
        :rtype: PVM_tools.bounding_region.BoundingRegion
        """
        im = self._preprocess(im)
        return self.tracker.track(im)

    def reset(self):
        """
        Reset the tracker

        :return:
        """
        self.tracker.reset()

    def get_heatmap(self, heatmap_name=None):
        """
        Get the original histogram backprojection heatmap

        :param heatmap_name: optional argument for tracker which may have multipe heatmaps
        :return: heatmap
        :rtype: numpy.ndarray
        """
        return self.tracker.get_heatmap()


class UVHistogramBackprojectionTracker(GenericVisionTracker):
    """
    This class is a variation of the histogram backprojection tracker for the U and V chromatic
    channels of the YUV space. The tracker takes an RGB image, converts it into a proper format and runs through
    a appropriately configured instance of the backprojection tracker.

    :param bins: histogram bins shape (8, 8)
    :type bins: tuple
    :param channels: selection of channels (0, 1)
    :type channels: tuple
    :param ranges: histogram ranges (0, 180, 0, 255)
    :type ranges: tuple
    """

    def __init__(self):
        self.tracker = BasicHistogramBackprojectionTracker(bins=(8, 8), channels=(1, 2), ranges=(0, 255, 0, 255))
        self.name = "UVTracker"

    def _preprocess(self, image):
        """
        Preprocess the image

        :param image:
        :return:
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    def _prime(self, im, bounding_box=None, **kwargs):
        """
        Prime the tracker

        :param im: priming image
        :type im: numpy.ndarray
        :param bounding_box: initial target bounding box
        :type bounding_box: PVM_tools.bounding_region.BoundingRegion
        :return: True if primed
        :rtype: bool
        """
        im = self._preprocess(im)
        self.tracker.prime(im, bounding_box=bounding_box)

    def _track(self, im):
        """
        Track

        :param im:  image
        :type im: numpy.ndarray
        :return: bounding region
        :rtype: PVM_tools.bounding_region.BoundingRegion
        """
        im = self._preprocess(im)
        return self.tracker.track(im)

    def reset(self):
        """
        Reset the tracker

        :return:
        """
        self.tracker.reset()

    def get_heatmap(self, heatmap_name=None):
        """
        Get the original histogram backprojection heatmap

        :param heatmap_name: optional argument for tracker which may have multipe heatmaps
        :return: heatmap
        :rtype: numpy.ndarray
        """
        return self.tracker.get_heatmap()
