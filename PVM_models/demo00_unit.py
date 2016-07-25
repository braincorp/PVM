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
import PVM_framework.AbstractExecutionUnit as AbstractExecutionUnit
import PVM_framework.fast_routines as fast_routines


class ExecutionUnit(AbstractExecutionUnit.ExecutionUnit):

    @classmethod
    def execution_steps(cls):
        """
        The method needs to have sufficient number of execute methods
        """
        return 1  # because there is only execute0 implemented

    def __init__(self, parameters):
        self.arena = parameters['arena']
        self.block = parameters['block']

    @staticmethod
    def generate_missing_parameters(parameters):
        """
        This method can be called to generate all the missing dictionary parameters when all
        the other relevant variables are known. Leave empty if there is nothing more to generate
        :return:
        """
        return parameters

    def execute0(self):
        """
        This method does the actual execution.
        """
        # Using cythonized code, runs much faster
        fast_routines.flip_some_bytes(100, self.arena.view(np.ndarray), self.block.view(np.ndarray))

    def cleanup(self):
        """
        This needs to be implemented but may be empty if the entire state is
        always kept in the dictionary elements (external). If some internal state exists,
        here is the place to copy it back to an external variable.
        """
        pass
