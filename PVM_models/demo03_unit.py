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
import PVM_framework.SharedArray as SharedArray
import PVM_framework.MLP as MLP


class ExecutionUnit(AbstractExecutionUnit.ExecutionUnit):

    @classmethod
    def execution_steps(cls):
        """
        The method needs to have sufficient number of execute methods
        :return:
        """
        return 2  # because there is execute0 and execute1 implemented

    def __init__(self, parameters):
        self.array = parameters['array'].view(np.ndarray)
        self.divider = parameters['array'].shape[0]-(2*parameters['array'].shape[0])/3
        self.predicted_array = parameters['predicted_array'].view(np.ndarray)
        self.block = parameters['block'].view(np.ndarray)
        self.output_block = parameters['output_block']
        self.context_blocks = parameters['context_blocks']
        self.MLP = MLP.MLP(parameters["MLP_parameters"])

    @staticmethod
    def generate_missing_parameters(parameters):
        """
        This method can be called to generate all the missing dictionary parameters when all
        the other relevant variables are known. Leave empty if there is nothing more to generate
        """
        ninputs = ((2*parameters['array'].shape[0])/3)*parameters['block'][2]*parameters['block'][3]*3
        for block in parameters["context_blocks"]:
            ninputs += np.prod(block.shape)
        nhidden = parameters['internal_rep_size']
        noutputs = (parameters['array'].shape[0]-(2*parameters['array'].shape[0])/3)*parameters['block'][2]*parameters['block'][3]*3
        parameters["MLP_parameters"] = {}
        parameters["MLP_parameters"]['layers'] = [
            {'activation': SharedArray.SharedNumpyArray((ninputs+1), np.float),
             'error': SharedArray.SharedNumpyArray((ninputs+1), np.float),
             'delta': SharedArray.SharedNumpyArray((ninputs+1), np.float)
             },
            {'activation': SharedArray.SharedNumpyArray((nhidden+1), np.float),
             'error': SharedArray.SharedNumpyArray((nhidden+1), np.float),
             'delta': SharedArray.SharedNumpyArray((nhidden+1), np.float)
             },
            {'activation': SharedArray.SharedNumpyArray((noutputs+1), np.float),
             'error': SharedArray.SharedNumpyArray((noutputs+1), np.float),
             'delta': SharedArray.SharedNumpyArray((noutputs+1), np.float)
             },
        ]
        parameters["MLP_parameters"]['beta'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["MLP_parameters"]['beta'][0] = 1.0
        parameters["MLP_parameters"]['learning_rate'] = parameters['learning_rate']
        parameters["MLP_parameters"]['momentum'] = parameters['momentum']
        parameters["MLP_parameters"]['mse'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["MLP_parameters"]['weights'] = [
            MLP.initialize_weights(SharedArray.SharedNumpyArray((ninputs+1, nhidden), np.float)),
            MLP.initialize_weights(SharedArray.SharedNumpyArray((nhidden+1, noutputs), np.float))
        ]

    def execute0(self):
        """
        This method does the actual execution.
        """
        input = self.array[self.divider:, self.block[0]:self.block[0]+self.block[2], self.block[1]:self.block[1]+self.block[3]].flatten()
        for block in self.context_blocks:
            input = np.append(input, block.flatten())
        output = self.array[0:self.divider, self.block[0]:self.block[0]+self.block[2], self.block[1]:self.block[1]+self.block[3]].flatten()
        predicted = self.MLP.train(input.astype(np.float)/255, output.astype(np.float)/255)
        predicted = 255*predicted.reshape((self.divider, self.block[2], self.block[3], 3))
        self.predicted_array[:, self.block[0]:self.block[0]+self.block[2], self.block[1]:self.block[1]+self.block[3]] = predicted.astype(np.uint8)

    def execute1(self):
        # This has to happen after a barrier so that all the data is consistent
        self.output_block[:] = self.MLP.get_activation(1)

    def cleanup(self):
        """
        This needs to be implemented but may be empty if the entire state is
        always kept in the dictionary elements (external). If some internal state exists,
        here is the place to copy it back to an external variable.
        """
        pass
