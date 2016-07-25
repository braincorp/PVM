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
    """

    Does everything the PVM unit does, but at the end just copies its previous input to the output.
    Used for testing and debugging.

    """
    SIGNAL_BLOCK_CONTENTS = set(["block: Raw signal block",
                                 "delta: Delta extracted from associator",
                                 "pred_block: Predicted block, output",
                                 "past_block: Past block, input",
                                 "dblock: Derivative of raw signal block",
                                 "iblock: Integral of raw signal block",
                                 "pred_block_local: Clean local copy of predicted block"])

    UNSUPERVISED_SIGNAL_INPUTS = set(["block: Raw signal block",
                                      "dblock: Derivative of raw signal block",
                                      "iblock: Integral of raw signal block",
                                      "e_block: Error in prediction"])

    UNSUPERVISED_CONTEXT_INPUTS = set(["block: Raw feedback or context input Block",
                                       "delta: Delta of the context inputs [currently NOT computed or used]",
                                       "factor: Factor that scales this feedback or context input"])

    SUPERVISED_TASK_OUTPUTS = set(["block: Target heatmap",
                                   "delta: Prediction error of heatmap",
                                   "pblock: Prediction of heatmap",
                                   "pdblock: Prediction of derivative of heatmap"])

    @classmethod
    def execution_steps(cls):
        """
        The method needs to have sufficient number of execute methods
        :return:
        """
        return 3  # because there is execute0, 1 and 2 implemented

    def open_views_into_shmem(self, shared_memory, key, num_items):
        return map(lambda x: tuple([x[i].view(np.ndarray) for i in range(num_items)]), shared_memory[key])

    def __init__(self, parameters):
        # Mapping SharedArray items through .view(np.ndarray) improved the access time
        self.signal_blocks = self.open_views_into_shmem(parameters, "signal_blocks", len(ExecutionUnit.SIGNAL_BLOCK_CONTENTS))
        self.predicted_blocks = self.open_views_into_shmem(parameters, "predicted_blocks", len(ExecutionUnit.SUPERVISED_TASK_OUTPUTS))
        self.context_blocks = self.open_views_into_shmem(parameters, "context_blocks", len(ExecutionUnit.UNSUPERVISED_CONTEXT_INPUTS))
        self.output_block = parameters['output_block'].view(np.ndarray)  # Will appear in other places as either input or context

        # The perceptron
        self.MLP = MLP.MLP(parameters["MLP_parameters"])
        self.MLP_1 = MLP.MLP(parameters["MLP_parameters_additional"])  # This is a "task" supervised MLP (e.g., tracker)
        self.learning_rate = parameters["MLP_parameters"]['learning_rate']
        self.learning_rate_1 = parameters["MLP_parameters_additional"]['learning_rate']
        self.layers = parameters["MLP_parameters"]['layers']
        self.layers_1 = parameters["MLP_parameters_additional"]['layers']  # These are the "task" supervised MLP layers
        self.tau = parameters['tau']  # Tau is the integration constant for the signal integral

        # Input buffers
        self.ninputs = 0
        self.npredictions = 0
        self.npinputs = 0
        self.ncontexts = 0
        for (block, delta, pred_block, past_block, dblock, iblock, pred_block_local) in self.signal_blocks:
            self.ninputs += len(ExecutionUnit.UNSUPERVISED_SIGNAL_INPUTS) * np.prod(block.shape)
            self.npredictions += np.prod(block.shape)
        for (block, delta, pblock, pdblock) in self.predicted_blocks:
            self.npinputs += np.prod(block.shape)
        self.inputs_t = np.zeros((self.ninputs,))
        self.predictions_t = np.zeros((self.npredictions,))
        self.pinputs_t = np.zeros((self.npinputs,))

        # Context buffers
        for (block, delta, factor) in self.context_blocks:
            self.ncontexts += np.prod(block.shape)
        self.contexts_t_1 = np.zeros((self.ncontexts,))

        # Buffer for storing the averaged out deltas
        self.output_layer = len(self.layers)-2  # Becuse the "output" of this Unit comes from its hidden layer
        self.output_length = np.prod(self.output_block.shape)

        # additional flags
        if 'autoencoder' in parameters.keys() and parameters['autoencoder']:
            self.autoencoder = True
        else:
            self.autoencoder = False
        if 'complex' in parameters.keys() and parameters['complex']:
            self.complex = True
        else:
            self.complex = False

    @staticmethod
    def generate_missing_parameters(parameters, complex_unit=False, complex_unit_extra_layer=False, polynomial=False, autoencoder=False):
        """
        This method can be called to generate all the missing dictionary parameters when all
        the other relevant variables are known. Leave empty if there is nothing more to generate.
        When complex_unit is False, a standard 3-layer MLP is used.
        When complex_unit is True, an MLP with additional hidden layers is used.

        There needs to be no return value, the method leaves a side effect by modifying the perameters dict.

        :param parameters: parameter dictionary
        :type parameters: dict
        """
        nhidden = np.prod(parameters['output_block'].shape)
        ntohidden = np.prod(parameters['output_block'].shape)
        ninputs = 0
        noutputs = 0
        ncontext = 0
        for (block, delta, pred_block, past_block, dblock, iblock, pred_block_local) in parameters['signal_blocks']:
            ninputs += np.prod(block.shape)*len(ExecutionUnit.UNSUPERVISED_SIGNAL_INPUTS)
        for (block, delta, factor) in parameters['context_blocks']:
            # ninputs += np.prod(block.shape)
            ncontext += np.prod(block.shape)
        for (block, delta, pred_block, past_block, dblock, iblock, pred_block_local) in parameters['signal_blocks']:
            noutputs += np.prod(block.shape)
        nadditional = 0
        for (block, delta, pblock, pdblock) in parameters['predicted_blocks']:
            nadditional += np.prod(block.shape)
        nmiddle=nhidden*2
        parameters["MLP_parameters"] = {}
        parameters["MLP_parameters_additional"] = {}
        if complex_unit:
            mlp_layers = [{'activation': SharedArray.SharedNumpyArray((ninputs+1), np.float),
                           'error': SharedArray.SharedNumpyArray((ninputs+1), np.float),
                           'delta': SharedArray.SharedNumpyArray((ninputs+1), np.float)
                           },
                          {'activation': SharedArray.SharedNumpyArray((nmiddle+ncontext+1), np.float),
                           'error': SharedArray.SharedNumpyArray((nmiddle+ncontext+1), np.float),
                           'delta': SharedArray.SharedNumpyArray((nmiddle+ncontext+1), np.float)
                           },
                          {'activation': SharedArray.SharedNumpyArray((nhidden+1), np.float),
                           'error': SharedArray.SharedNumpyArray((nhidden+1), np.float),
                           'delta': SharedArray.SharedNumpyArray((nhidden+1), np.float)
                           }]

            if complex_unit_extra_layer:
                mlp_layers.append({'activation': SharedArray.SharedNumpyArray((nhidden+1), np.float),
                                   'error': SharedArray.SharedNumpyArray((nhidden+1), np.float),
                                   'delta': SharedArray.SharedNumpyArray((nhidden+1), np.float)
                                   })

            mlp_layers.append({'activation': SharedArray.SharedNumpyArray((noutputs+1), np.float),
                               'error': SharedArray.SharedNumpyArray((noutputs+1), np.float),
                               'delta': SharedArray.SharedNumpyArray((noutputs+1), np.float)
                               })

            parameters["MLP_parameters"]['layers'] = mlp_layers
            parameters["MLP_parameters"]['beta'] = SharedArray.SharedNumpyArray((1,), np.float)
            parameters["MLP_parameters"]['beta'][0] = 1.0
            parameters["MLP_parameters"]['learning_rate'] = parameters['learning_rate']
            parameters["MLP_parameters"]['momentum'] = parameters['momentum']
            parameters["MLP_parameters"]['mse'] = SharedArray.SharedNumpyArray((1,), np.float)
            mlp_weights = [MLP.MLP.initialize_weights(SharedArray.SharedNumpyArray((ninputs+1, nmiddle), np.float)),
                           MLP.MLP.initialize_weights(SharedArray.SharedNumpyArray((nmiddle+ncontext+1, nhidden), np.float))]
            if complex_unit_extra_layer:
                mlp_weights.append(MLP.MLP.initialize_weights(SharedArray.SharedNumpyArray((nhidden+1, nhidden), np.float)))
            mlp_weights.append(MLP.MLP.initialize_weights(SharedArray.SharedNumpyArray((nhidden+1, noutputs), np.float)))
            parameters["MLP_parameters"]['weights'] = mlp_weights
            parameters["complex"] = True
        else:  # Simple MLP Unit (not complex unit)
            parameters["MLP_parameters"]['layers'] = [
                {'activation': SharedArray.SharedNumpyArray((ninputs+ncontext+1), np.float),
                 'error': SharedArray.SharedNumpyArray((ninputs+ncontext+1), np.float),
                 'delta': SharedArray.SharedNumpyArray((ninputs+ncontext+1), np.float)
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
                MLP.MLP.initialize_weights(SharedArray.SharedNumpyArray((ninputs+ncontext+1, nhidden), np.float)),
                MLP.MLP.initialize_weights(SharedArray.SharedNumpyArray((nhidden+1, noutputs), np.float)),
            ]

        parameters["MLP_parameters_additional"]['layers'] = [
            {'activation': SharedArray.SharedNumpyArray((nhidden+1), np.float),
             'error': SharedArray.SharedNumpyArray((nhidden+1), np.float),
             'delta': SharedArray.SharedNumpyArray((nhidden+1), np.float)
             },
            {'activation': SharedArray.SharedNumpyArray((nadditional+1), np.float),
             'error': SharedArray.SharedNumpyArray((nadditional+1), np.float),
             'delta': SharedArray.SharedNumpyArray((nadditional+1), np.float)
             },
        ]
        parameters["MLP_parameters_additional"]['beta'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["MLP_parameters_additional"]['beta'][0] = 1.0
        parameters["MLP_parameters_additional"]['learning_rate'] = parameters['additional_learning_rate']
        parameters["MLP_parameters_additional"]['momentum'] = parameters['momentum']
        parameters["MLP_parameters_additional"]['mse'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["MLP_parameters_additional"]['weights'] = [
            MLP.MLP.initialize_weights(SharedArray.SharedNumpyArray((nhidden+1, nadditional), np.float)),
        ]
        if polynomial:
            parameters["MLP_parameters"]['polynomial'] = True
            parameters["MLP_parameters_additional"]['polynomial'] = True
        if autoencoder:
            parameters['autoencoder'] = True

    def execute0(self):
        """
        Make predictions based on available information and publish them.

        """
        # Prepare the input vector
        if not self.complex:
            self.layers[0]['activation'][:-1] = np.concatenate((self.inputs_t, self.contexts_t_1))
        else:
            self.layers[0]['activation'][:-1] = self.inputs_t
            self.layers[1]['activation'][self.output_length*2:-1] = self.contexts_t_1
        # Make prediction
        self.MLP.mlp_forward()
        self.layers_1[0]['activation'][:-1] = self.layers[self.output_layer]['activation'][:-1]
        self.MLP_1.mlp_forward()
        # Update the output
        self.output_block[:] = (self.layers[self.output_layer]['activation'][:self.output_length]).reshape(self.output_block.shape)
        # Publish predictions (mainly for debugging and inspection)
        i = 0
        for (block, delta, pred_block, past_block, dblock, iblock, pred_block_local) in self.signal_blocks:
            l = np.prod(pred_block.shape)
            pred_block[:] = (self.inputs_t[i:i+l]).reshape(pred_block.shape)
            pred_block_local[:] = (self.inputs_t[i:i+l]).reshape(pred_block_local.shape)
            i += 4*l
        i = 0
        for (block, delta, predicted_block, predicted_d_block) in self.predicted_blocks:
            l = np.prod(predicted_block.shape)
            predicted_block[:] = (self.inputs_t[i:i+l]).reshape(predicted_block.shape)
            i += l

    def execute1(self):
        """
        Collect new signals, perform association, prepare for additional learning
        """
        # Collect the current signal
        i = 0
        j = 0
        for (block, delta, pred_block, past_block, dblock, iblock, pred_block_local) in self.signal_blocks:
            l = np.prod(block.shape)
            # Calculate signal features
            e_block = 0.5*(block-pred_block_local)+0.5  # Rescale so it could possibly be predicted by sigmoid units
            dblock[:] = 0.5*(block-past_block)+0.5  # Rescale so it could possibly be predicted by sigmoid units
            past_block[:] = block
            iblock[:] = self.tau[0] * iblock + (1-self.tau[0]) * block
            # Proportional signal
            self.inputs_t[i:i + l] = block.flatten()
            # Derivative signal
            self.inputs_t[i + l:i + 2 * l] = dblock.flatten()
            # Integral signal
            self.inputs_t[i + 2 * l:i + 3 * l] = iblock.flatten()
            # Prediction Error signal
            self.inputs_t[i + 3 * l:i + 4 * l] = e_block.flatten()
            i += len(ExecutionUnit.UNSUPERVISED_SIGNAL_INPUTS)*l
            # For autoencoder we keep the previous signal
            if not self.autoencoder:
                self.predictions_t[j:j+l] = block.flatten()
                j += l

        i = 0
        for (block, delta, pblock, pdblock) in self.predicted_blocks:
            l = np.prod(block.shape)
            self.pinputs_t[i:i + l] = block.flatten()
            i += l

        if self.learning_rate[0] != 0:
            # Calculate the prediction error
            error_x = self.predictions_t - self.layers[-1]['activation'][:np.prod(self.predictions_t.shape)]
            # Make the association
            self.MLP.train2(error=error_x)
            # Extract the new deltas for input blocks
            i = 0
            for (block, delta, pred_block, past_block, dblock, iblock, pred_block_local) in self.signal_blocks:
                l = np.prod(block.shape)
                delta[:] = self.layers[0]['delta'][i:i + l].reshape(delta.shape)
                i += len(ExecutionUnit.UNSUPERVISED_SIGNAL_INPUTS)*l
        if self.learning_rate_1[0] != 0:
            error_p = self.pinputs_t - self.layers_1[-1]['activation'][:-1]
            self.MLP_1.train2(error=error_p)
        # For autoencoder new signal is collected after the training is done
        if self.autoencoder:
            j = 0
            for (block, delta, pred_block, past_block, dblock, iblock, pred_block_local) in self.signal_blocks:
                l = np.prod(block.shape)
                self.predictions_t[j:j+l] = block.flatten()
                j += l

    def execute2(self):
        """
        Gather the context signal in preparation for the next step.
        Some context blocks get their information from units lateral to this one,
        others get their information from units above this one.
        The "factor" is either the context_factor_lateral or context_factor_feedback
        """
        # Prepare information for the next step
        # collect the new context
        i = 0
        for (block, delta, factor) in self.context_blocks:
            l = np.prod(block.shape)
            self.contexts_t_1[i:i+l] = factor[0]*block.flatten()
            i += l

    def cleanup(self):
        """
        This needs to be implemented but may be empty if the entire state is
        always kept in the dictionary elements (external). If some internal state exists,
        here is the place to copy it back to an external variable.
        """
        pass
