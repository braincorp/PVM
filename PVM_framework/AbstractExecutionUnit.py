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

import abc


class ExecutionUnit(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def execution_steps(cls):
        """
        The method needs to have sufficient number of execute methods
        Note, this needs to be extended as a class method
        """
        pass

    @abc.abstractmethod
    def __init__(self, parameters):
        pass

    @abc.abstractmethod
    def generate_missing_parameters(parameters, options):
        """
        This method can be called to generate all the missing dictionary parameters when all
        the other relevant viariables are known. Needs to be extended as a static method.
        :return:
        """
        pass

    @abc.abstractmethod
    def execute0(self):
        """
        This method will do the execution. It is nescessary, but you can implement more, like execute1 and so on.
        Their execution will be interleaved with a barrier.
        :return:
        """
        pass

    @abc.abstractmethod
    def cleanup(self):
        """
        WIll be called before the simulation is finished. Allows to copy back any remaining state into
        the share memory objects
        :return:
        """
        pass
