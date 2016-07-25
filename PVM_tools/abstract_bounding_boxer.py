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
This module contains the abstract bounding boxer class.

Import this module as:
::

    import PVM_tools.abstract_bounding_boxer

or:
::

    from PVM_tools.abstract_bounding_boxer import AbstractBoundingBoxer

"""

from abc import ABCMeta, abstractmethod


class AbstractBoundingBoxer(object):
    """
    This is an abstract class defining all the methods a bounding boxer has to implement.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_current_bounding_box(self, bounding_region):
        """
        Sets the current bounding region

        :param bounding_region: an object defining the region of interest
        :type bounding_region: PVM_tools.bounding_region.BoundingRegion

        .. note::
            This is an abstract method, each bounding boxer needs to implement it
        """

    @abstractmethod
    def process(self, heatmap, previous_bb=None):
        """
        Processes new heatmap, tries to find the next bounding box

        :param heatmap: new probability/likelihood heatmap defining regions of interest
        :type heatmap: numpy.ndarray
        :param previous_bb: optional previous bounding box
        :type previous_bb: PVM_tools.bounding_region.BoundingRegion
        :return:
        """

    @abstractmethod
    def reset(self):
        """
        Reset the state of the bounding boxer
        :return:
        """
