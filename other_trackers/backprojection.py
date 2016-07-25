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


class ColorHistogramBackProjection(object):

    '''
    Takes a color histogram from a sample image and "back-projects"
    the histogram, using it as a color filter for the image.

    Outputs a float image in range (0, 1), indicating membership
    to color histogram.

    :param n_bins: number of bins on each channel
    :type n_bins: 3-tuple
    :param channels: the image channels to draw the historgram from.
    :type channels: 3-tuple
    :param ranges: min and max for each channel
    :type ranges: 3-tuple
    :param use_background: if True, the histogram values of the target bounding box
        will be divided by the whole image histogram values, so that the
        backprojection values represent p(target|bin)
    :type use_background: Boolean

    .. automethod:: _update_hist

    '''

    def __init__(self, n_bins=(16, 16, 16), channels=(0, 1, 2), ranges=(0, 256, 0, 256, 0, 256), use_background=True):
        self.n_bins = n_bins
        self._ranges = ranges
        self._channels = channels
        self.use_background = use_background
        self.hist = None
        self.raw_hist_target = None
        self.raw_hist_image = None
        self.out = None

    def calculateHistogram(self, image, bb=None):
        '''
        :param image: input image
        :type image: numpy.ndarray
        :param bb: bounding box
        :type bb: numpy.ndarray
        If bb (=(x, y, w, h)) is not None, the histogram is taken from the bounded
        part of the image. If use_background is True, bb cannot be None

        .. note::
            The bounding box here is given in pixel coordinates!
        '''
        assert not (self.use_background and bb is None), 'If using background, bb must be provided'

        if bb is not None:
            x, y, w, h = bb
            target = image[y:y + h, x:x + w]
        else:
            target = image
        hist_target = cv2.calcHist([target], channels=self._channels, mask=None, histSize=self.n_bins, ranges=self._ranges)
        if self.raw_hist_target is None:
            self.raw_hist_target = hist_target
        else:
            self.raw_hist_target += hist_target

        if self.use_background:
            hist_image = cv2.calcHist([image], channels=self._channels, mask=None, histSize=self.n_bins, ranges=self._ranges)
            if self.raw_hist_image is None:
                self.raw_hist_image = hist_image
            else:
                self.raw_hist_image += hist_image
        self._update_hist()

    def _update_hist(self):
        """
        Internal method to calculate the actual histogram
        """
        if self.use_background:
            if self.raw_hist_image is not None:
                self.hist = self.raw_hist_target / (self.raw_hist_image + 1.)
            else:
                self.hist = self.raw_hist_target.copy()
        else:
            self.hist = cv2.normalize(self.raw_hist_target, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX)

    def get_color_histogram(self):
        """
        Returns the current histogram
        :return: current histogram
        :rtype: numpy.ndarray of type float32
        """
        return self.hist

    def add_color_histogram(self, hist):
        """
        Additive modification of the current histogram
        :param hist: a numpy array in the same shape as the current histogram
        :type hist: numpy.ndarray
        """
        if self.raw_hist_target is None:
            self.raw_hist_target = hist.copy()
        else:
            self.raw_hist_target += hist
        self._update_hist()

    def subtract_color_histogram(self, hist):
        """
        Subtractive modification of the current histogram
        :param hist: a numpy array in the same shape as the current histogram
        :type hist: numpy.ndarray
        """
        if self.raw_hist_target is not None:
            self.raw_hist_target -= hist
            self.raw_hist_target[self.raw_hist_target < 0] = 0
        self._update_hist()

    def reset(self):
        """
        Reset the object
        """
        self.raw_hist_target = None
        self.raw_hist_image = None
        self.hist = None

    def calculate(self, image):
        """
        Calculate the histogram packprojection on the given image using
        the currently stored histogram
        :param image: numpy array with appropriate number of channels
        :type image: numpy.ndarray
        :return: a numpy array of type float32 containing the histogram backprojection
        :rtype: numpy.ndarray of type float32
        """
        if self.hist is None:
            return np.zeros(image.shape[:-1], dtype=np.float32)

        out = cv2.calcBackProject([image.astype('float32')], channels=self._channels, hist=self.hist, ranges=self._ranges, scale=1)
        self.out = out
        return self.out
