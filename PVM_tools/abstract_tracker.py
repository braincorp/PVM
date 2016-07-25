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
Abstract tracker
----------------

This module contains the basic definition of abstract or semi abstract types that any vision
tracker should inherit. The main functionality of a tracker in encapsulated by two methods:

    1. prime - used to inform the tracker which object or area its supposed to track
    2. track - once primed, given an image, return the bounding box


This module contains the abstract bounding boxer class.

Import this module as:
::

    import PVM_tools.abstract_tracker

or:
::

    from PVM_tools.abstract_tracker import AbstractVisionTracker, GenericVisionTracker

"""

from abc import ABCMeta, abstractmethod


class AbstractVisionTracker(object):
    """
    This is an abstract class defining all the methods a vision tracker has to implement.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def prime(self, im=None, bounding_box=None, **kwargs):
        """
        :param im: Image as a numpy array or extension thereof
        :type im: numpy.ndarray
        :param bounding_box: instance of the bounding region class
        :type bounding_box: PVM_tools.bounding_region.BoundingRegion
        :param kwargs: additional implementation specific parameters

        Prime tracker on image and (optional) bounding box

        If im is None, priming will be done on the last tracked image
        kwargs can be passed on to particular implementations

        .. note::
            This is an abstract method and needs to be implemented by inheriting classes.
        """

    @abstractmethod
    def track(self, im):
        """
        :param im: numpy array representing an image
        :type im: numpy.ndarray
        :returns: bounding_box
        :rtype: PVM_tools.bounding_region.BoundingRegion

        Track on a given image

        .. note::
            This is an abstract method and needs to be implemented by inheriting classes.
        """

    @abstractmethod
    def get_bbox(self):
        """
        :returns: bounding_box
        :rtype: PVM_tools.bounding_region.BoundingRegion

        Return current bounding box of the target

        .. note::
            This is an abstract method and needs to be implemented by inheriting classes.
        """

    @abstractmethod
    def get_heatmap(self, heatmap_name=None):
        """
        :param heatmap_name: string object
        :type heatmap_name: string
        :returns: heatmap representing the probability of where the target object is
        :rtype: numpy.ndarray

        Return an internal heatmap for current tracker state.
        There may be several heatmaps available that can be requested
        specifying 'heatmap_name' parameter

        .. note::
            This is an abstract method and needs to be implemented by inheriting classes.
        """


class GenericVisionTracker(AbstractVisionTracker):
    """
    Baseclass that describes default behaviors of most of the trackers.


    .. automethod:: _prime
    .. automethod:: _track
    """
    @abstractmethod
    def _prime(self, im, bounding_box=None, **kwargs):
        """
        The actual internal priming method called by prime. Trackers inheriting
        from GenericVisionTracker class need to implement this method.

        .. note::
            This is an abstract method and needs to be implemented by inheriting classes.
        """

    def prime(self, im=None, bounding_box=None, **kwargs):
        """
        :param im: Image as a numpy array or extension thereof
        :type im: numpy.ndarray
        :param bounding_box: bounding box (or in general any bounding region) of the target
        :type bounding_box: PVM_tools.bounding_region.BoundingRegion
        :param kwargs: additional implementation specific parameters

        Prime tracker on an image and (optional) an bounding box
        bounding box as an instance of PVM_tools.bounding_region.bounding_region class

        If im is None, priming will be done on the last tracked image
        kwargs can be passed on to particular implementations

        .. note::
            Many trackers will require the bounding region object. Others may however
            be able to prime automatically e.g. base on the saliency of the object or prime on the object in
            the center of view etc.
        """
        if im is None:
            if hasattr(self, '_last_tracked_im'):
                im = self._last_tracked_im
            else:
                raise Exception('Trying to prime on last tracked image, but no images were tracked yet')
        return self._prime(im, bounding_box, **kwargs)

    @abstractmethod
    def _track(self, im):
        """
        The actual internal tracking method. Trackers inheriting from GenericVisionTracker
        need to implement this method.

        .. note::
            This is an abstract method and needs to be implemented by inheriting classes.

        """

    def track(self, im):
        """
        Track the object of interest in the image.

        :param im: numpy array representing an image
        :type im: numpy.ndarray
        :returns: bounding_box
        :rtype: PVM_tools.bounding_region.BoundingRegion
        """
        self.bbox = self._track(im)
        self._last_tracked_im = im
        return self.bbox

    def get_bbox(self):
        """
        :returns: bounding_box
        :rtype: PVM_tools.bounding_region.BoundingRegion

        Return current bounding box of the target
        """
        if hasattr(self, 'bbox'):
            return self.bbox
        else:
            return None

    def get_heatmap(self, heatmap_name=None):
        """
        :param heatmap_name: string object
        :type heatmap_name: string
        :returns: heatmap representing the probability of where the target object is
        :rtype: numpy.ndarray

        Return an internal heatmap for current tracker state.
        There may be several heatmaps available that can be requested
        specifying 'heatmap_name' parameter

        This is default implementation - no heatmap available and None is returned.
        """
        return None

    def get_name(self):
        """
        Get the tracker name.

        :return: name of the tracker
        :rtype: str
        """
        return self.name

    def finish(self):
        """
        Call this method whenever the work with tracker has ended. This method will perform
        any necessary cleanup.

        .. note::
            This call should be considered a part of destruction, that is after finish the object will
            not be used again and will be released. E.g. the tracker benchmark will create the tracker and
            reset if before each movie, but finish will be called at the end.

            Another example would be a tracker that is executing on a multi core machine using e.g. future encoder
            framework would use this call to stop all the worker processes and debug threads and so on.

        """
        pass
