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
import copy
import logging


class ExecutionUnit(AbstractExecutionUnit.ExecutionUnit):
    """
    Unit architecture:

    [predicted block],[predicted block], ... ,[predicted pblock],[predicted pblock]
     \                                                                           /
      \                                                                         /
       \                                                                       /
         [ compressed features - output block                                 ]
        /                                                                      \
       /                                                                        \
      /                                                                          \
     [block][eblock][dblock][iblock],[block] ...,   [context block][context block]

     Features used by the unit to predict each block:
      - previous value of the block
      - error in previous prediction
      - temporal derivative of the block
      - integral of the block
      - ?

    """
    # signal_block, delta_block, prediction_t+1, prediction_t+2
    SIGNAL_BLOCK_CONTENTS = set(["signal_block: Raw signal block",
                                 "delta_block: Delta extracted from associator",
                                 "prediction_tp1: Predicted block at t+1",
                                 "prediction_tp2: Predicted block at t+2"])

    UNSUPERVISED_SIGNAL_INPUTS = set(["block: Raw signal block t-1"
                                      ])

    UNSUPERVISED_CONTEXT_INPUTS = set(["block: Raw feedback or context input Block",
                                       "delta: Delta of the context inputs [currently NOT computed or used]",
                                       "factor: Factor that scales this feedback or context input"])
    # Each block: teaching_signal, delta_block, readout_block
    SUPERVISED_TASK_OUTPUTS = set(["teaching_block: Target heatmap",
                                   "delta: Prediction error of heatmap",
                                   "readout_block: Prediction of heatmap"])

    @classmethod
    def execution_steps(cls):
        """
        The method needs to have sufficient number of execute methods
        :return:
        """
        return 3  # because there is execute0, 1 and 2 implemented

    def open_views_into_shmem(self, shared_memory, key, num_items):
        return map(lambda x: tuple([x[i].view(np.ndarray) for i in range(num_items)]), shared_memory[key])

    def open_views_into_shmem_single(self, shared_memory, key):
        return map(lambda x: x.view(np.ndarray), shared_memory[key])

    def __init__(self, parameters):
        # Mapping SharedArray items through .view(np.ndarray) improved the access time
        self.signal_blocks = self.open_views_into_shmem(parameters, "signal_blocks", len(ExecutionUnit.SIGNAL_BLOCK_CONTENTS))
        self.readout_blocks = self.open_views_into_shmem(parameters, "readout_blocks", len(ExecutionUnit.SUPERVISED_TASK_OUTPUTS))
        self.context_blocks = self.open_views_into_shmem(parameters, "context_blocks", len(ExecutionUnit.UNSUPERVISED_CONTEXT_INPUTS))
        self.derivative_blocks = self.open_views_into_shmem_single(parameters, "derivative_blocks")
        self.delta_blocks = self.open_views_into_shmem_single(parameters, "delta_blocks")
        self.integral_blocks = self.open_views_into_shmem_single(parameters, "integral_blocks")
        self.error_blocks = self.open_views_into_shmem_single(parameters, "error_blocks")
        self.internal_buffers = self.open_views_into_shmem(parameters, "internal_buffers", 3)
        self.output_block = parameters['output_block'].view(np.ndarray)  # Will appear in other places as either input or context
        self.output_min = parameters['output_min']
        self.output_max = parameters['output_max']
        self.avg_delta = parameters['avg_delta']
        self.internal_buffers = parameters['internal_buffers']
        self.flags = parameters['flags']
        # The perceptron
        self.MLP_internal_prediction = MLP.MLP(parameters["Primary_Predictor_params"])
        self.MLP_readout = MLP.MLP(parameters["Readout_Predictor_params"])  # This is a "task" supervised MLP (e.g., tracker)
        self.primary_learning_rate = parameters['primary_learning_rate']
        self.readout_learning_rate = parameters['readout_learning_rate']
        self.layers_internal_prediction = parameters["Primary_Predictor_params"]['layers']
        self.layers_readout = parameters["Readout_Predictor_params"]['layers']  # These are the "task" supervised MLP layers
        self.tau = parameters['tau']  # Tau is the integration constant for the signal integral
        # Operation parameters
        self.input_blocks_skip = 1
        self.output_blocks_skip = 1
        self.backpropagate_readout_error = False
        self.complex = False
        self.complex_context_in_second_layer = False
        self.use_global_backprop = False
        self.use_t_2_block = False
        self.use_derivative = False
        self.use_error = False
        self.use_integral = False
        self.predict_2_steps = False
        self.normalize = False
        if 'backpropagate_readout_error' in parameters.keys():
            self.normalize = parameters['backpropagate_readout_error']
        if 'normalize_output' in parameters.keys():
            self.normalize = parameters['normalize_output']
        if 'complex' in parameters.keys():
            self.complex = parameters['complex']
        if 'complex_context_in_second_layer' in parameters.keys():
            self.complex_context_in_second_layer = parameters['complex_context_in_second_layer']
        if 'use_derivative' in parameters.keys():
            self.use_derivative = parameters['use_derivative']
        if 'use_integral' in parameters.keys():
            self.use_integral = parameters['use_integral']
        if 'use_error' in parameters.keys():
            self.use_error = parameters['use_error']
        if 'use_t_2_block' in parameters.keys():
            self.use_t_2_block = parameters['use_t_2_block']
        if 'use_global_backprop' in parameters.keys():
            self.use_global_backprop = parameters['use_global_backprop']
        if 'autoencoder' in parameters.keys():
            self.autoencoder = parameters['autoencoder']
        if 'predict_2_steps' in parameters.keys():
            self.predict_2_steps = parameters['predict_2_steps']
        if self.use_derivative:
            self.input_blocks_skip += 1
        if self.use_integral:
            self.input_blocks_skip += 1
        if self.use_error:
            self.input_blocks_skip += 1
        if self.use_t_2_block:
            self.input_blocks_skip += 1
        if self.predict_2_steps:
            self.output_blocks_skip += 1
        # Input buffers
        self.ninputs = 0
        self.npredictions = 0
        self.npinputs = 0
        self.ncontexts = 0
        for (block, delta, pred_block1, pred_block2) in self.signal_blocks:
            self.ninputs += self.input_blocks_skip * np.prod(block.shape)
            self.npredictions += self.output_blocks_skip * np.prod(block.shape)
        for (teaching_block, delta, readout_block) in self.readout_blocks:
            self.npinputs += np.prod(teaching_block.shape)
        self.inputs_t = np.zeros((self.ninputs,))
        self.actual_signal_t = np.zeros((self.npredictions/self.output_blocks_skip, ))
        self.readout_training_signal = np.zeros((self.npinputs,))

        self.inputs_t_1 = np.zeros((self.ninputs,))
        self.actual_signal_t_1 = np.zeros((self.npredictions/self.output_blocks_skip, ))
        self.pinputs_t_1 = np.zeros((self.npinputs,))

        # Context buffers
        for (block, delta, factor) in self.context_blocks:
            self.ncontexts += np.prod(block.shape)
        self.contexts_t_1 = np.zeros((self.ncontexts,))
        self.output_layer = len(self.layers_internal_prediction)-2  # Becuse the "output" of this Unit comes from its hidden layer
        self.output_length = np.prod(self.output_block.shape)
        # Buffer for storing activations
        self.activations_buffer = []
        # additional flags
        if self.use_global_backprop:
            self.push_activation()
        if self.predict_2_steps:
            self.push_activation()
        # If two steps into the future are being predicted the training can only be done once all the data is avaliable
        # and therefore older activations need to be recovered.
        # Similarly in the case of the global backprop, the deltas computed on the current step correspnd to activations
        # from one step behind therefore again the state of the network needts to be rewinded one step back for training
        # In the case of using both global backprop and predicting two steps into the future things are a it more complicated.
        # First the training needts to be applied to the state of the network from one step behind. At this stage the deltas
        # are computed bu now these deltas belong to the state two steps behind, so for global backprop step the
        # state of the network needs to be rewinded one step further.
        # Since that is so, one needs to keep a queue of 3 states.

    @staticmethod
    def upgrade_to_ver_1(parameters):
        parameters['internal_buffers'] = []
        parameters['output_min'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])
        parameters['output_max'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])
        parameters['avg_delta'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])
        parameters['integral_blocks'] = []
        parameters['derivative_blocks'] = []
        parameters['error_blocks'] = []
        parameters['use_derivative'] = True
        parameters['use_integral'] = True
        parameters['use_error'] = True
        parameters['use_t_2_block'] = False
        parameters['predict_2_steps'] = False
        parameters['use_global_backprop'] = False
        parameters['normalize_output'] = False
        parameters["complex_context_in_second_layer"] = False
        for (block, delta, pred_block, pred_block2) in parameters['signal_blocks']:
            block01 = SharedArray.SharedNumpyArray_like(block)
            block02 = SharedArray.SharedNumpyArray_like(block)
            block03 = SharedArray.SharedNumpyArray_like(block)
            parameters['internal_buffers'].append((block01, block02, block03))
            parameters['derivative_blocks'].append(SharedArray.SharedNumpyArray_like(block))
            parameters['integral_blocks'].append(SharedArray.SharedNumpyArray_like(block))
            parameters['error_blocks'].append(SharedArray.SharedNumpyArray_like(block))
        if "complex" not in parameters.keys():
            parameters["complex"] = False
        if len(parameters["Primary_Predictor_params"]['layers']) == 4:
            parameters["complex"] = True
        if "autoencoder" not in parameters.keys():
            parameters["autoencoder"] = False
        if "readout_learning_rate" not in parameters.keys():
            parameters['readout_learning_rate'] = parameters["Primary_Predictor_params"]["learning_rate"]
        if "momentum" not in parameters.keys():
            parameters['momentum'] = parameters["Primary_Predictor_params"]["momentum"]
        nhidden = parameters["Primary_Predictor_params"]['layers'][-2]['activation'].shape[0]-1
        nreadout = 0
        nouputs = 0
        for (block, delta, pred_block, pred_block2) in parameters['signal_blocks']:
            nouputs += np.prod(block.shape)
        for (block, delta, pblock) in parameters['readout_blocks']:
            nreadout += np.prod(block.shape)
        if "Readout_Predictor_params" not in parameters.keys():
            parameters["Readout_Predictor_params"] = {}
            parameters["Readout_Predictor_params"]['layers'] = MLP.get_layers([nhidden+1, nreadout+1])
            parameters["Readout_Predictor_params"]['beta'] = SharedArray.SharedNumpyArray((1,), np.float)
            parameters["Readout_Predictor_params"]['beta'][0] = 1.0
            parameters["Readout_Predictor_params"]['learning_rate'] = parameters['readout_learning_rate']
            parameters["Readout_Predictor_params"]['momentum'] = parameters['momentum']
            parameters["Readout_Predictor_params"]['mse'] = SharedArray.SharedNumpyArray((1,), np.float)
            parameters["Readout_Predictor_params"]['weights'] = MLP.get_weights(parameters["Readout_Predictor_params"]['layers'])
            parameters["Readout_Predictor_params"]['weights'][0][:] = parameters["Primary_Predictor_params"]['weights'][-1][:, nouputs:]
            old_weight_matrix = parameters["Primary_Predictor_params"]['weights'][-1]
            parameters["Primary_Predictor_params"]['weights'][-1] = SharedArray.SharedNumpyArray((nhidden+1, nouputs), np.float)
            parameters["Primary_Predictor_params"]['weights'][-1][:] = old_weight_matrix[:, :nouputs]
            parameters["Primary_Predictor_params"]['layers'][-1] = {'activation': SharedArray.SharedNumpyArray(nouputs, np.float),
                                                                    'error': SharedArray.SharedNumpyArray(nouputs, np.float),
                                                                    'delta': SharedArray.SharedNumpyArray(nouputs, np.float)
                                                                    }
            parameters['backpropagate_readout_error'] = True

    @staticmethod
    def generate_missing_parameters(parameters, options):
        """
        This method can be called to generate all the missing dictionary parameters when all
        the other relevant variables are known. Leave empty if there is nothing more to generate.
        When complex_unit is False, a standard 3-layer MLP is used.
        When complex_unit is True, an MLP with additional hidden layers is used.

        There needs to be no return value, the method leaves a side effect by modifying the perameters dict.

        :param parameters: parameter dictionary
        :type parameters: dict
        """
        complex_unit = options['unit_type'] == "complex"
        polynomial = options['polynomial'] == '1'
        autoencoder = options['autoencoder'] == '1'
        use_t_2_block = options['use_t_minus_2_block'] == '1'
        use_derivative = options['use_derivative'] == '1'
        use_integral = options['use_integral'] == '1'
        use_error = options['use_error'] == '1'
        predict_2_steps = options['predict_two_steps'] == '1'
        use_global_backprop = options['use_global_backprop'] == '1'
        complex_context_in_second_layer = options['feed_context_in_complex_layer'] == '1'
        parameters['normalize_output'] = options["normalize_output"] == "1"
        parameters['backpropagate_readout_error'] = options["backpropagate_readout_error"] == "1"

        nhidden = np.prod(parameters['output_block'].shape)
        parameters['output_min'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])
        parameters['output_max'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])+1
        parameters['avg_delta'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])
        ninputs = 0
        noutputs = 0
        ncontext = 0
        # Any additional memory buffers needed in the operation of the unit
        parameters['internal_buffers'] = []
        parameters['integral_blocks'] = []
        parameters['derivative_blocks'] = []
        parameters['error_blocks'] = []
        parameters['use_derivative'] = use_derivative
        parameters['use_integral'] = use_integral
        parameters['use_error'] = use_error
        parameters['use_t_2_block'] = use_t_2_block
        parameters['predict_2_steps'] = predict_2_steps
        for (block, delta, pred_block, pred_block2) in parameters['signal_blocks']:
            block01 = SharedArray.SharedNumpyArray_like(block)
            block02 = SharedArray.SharedNumpyArray_like(block)
            block03 = SharedArray.SharedNumpyArray_like(block)
            parameters['internal_buffers'].append((block01, block02, block03))
            if use_derivative:
                parameters['derivative_blocks'].append(SharedArray.SharedNumpyArray_like(block))
            if use_integral:
                parameters['integral_blocks'].append(SharedArray.SharedNumpyArray_like(block))
            if use_error:
                parameters['error_blocks'].append(SharedArray.SharedNumpyArray_like(block))

        input_block_features = 1
        output_predictions = 1
        if use_derivative:
            input_block_features += 1
        if use_integral:
            input_block_features += 1
        if use_error:
            input_block_features += 1
        if use_t_2_block:
            input_block_features += 1
        if predict_2_steps:
            output_predictions += 1

        for (block, delta, pred_block, pred_block2) in parameters['signal_blocks']:
            ninputs += np.prod(block.shape) * input_block_features
        for (block, delta, factor) in parameters['context_blocks']:
            ncontext += np.prod(block.shape)
        for (block, delta, pred_block, pred_block2) in parameters['signal_blocks']:
            noutputs += np.prod(block.shape) * output_predictions

        nreadout = 0
        for (block, delta, pblock) in parameters['readout_blocks']:
            nreadout += np.prod(block.shape)
        parameters["Primary_Predictor_params"] = {}
        parameters["Readout_Predictor_params"] = {}
        if complex_unit and complex_context_in_second_layer:  # 4 layer perceptron
            parameters["Primary_Predictor_params"]['layers'] = MLP.get_layers([ninputs+1, 2*nhidden+ncontext+1, nhidden+1, noutputs+1])
        elif complex_unit:
            parameters["Primary_Predictor_params"]['layers'] = MLP.get_layers([ninputs+ncontext+1, 2*nhidden+1, nhidden+1, noutputs+1])
        else:  # 3 layer perceptron Simple MLP Unit (not complex unit)
            parameters["Primary_Predictor_params"]['layers'] = MLP.get_layers([ninputs+ncontext+1, nhidden+1, noutputs+1])

        parameters["Primary_Predictor_params"]['beta'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Primary_Predictor_params"]['beta'][0] = 1.0
        parameters["Primary_Predictor_params"]['learning_rate'] = parameters['primary_learning_rate']
        parameters["Primary_Predictor_params"]['momentum'] = parameters['momentum']
        parameters["Primary_Predictor_params"]['mse'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Primary_Predictor_params"]['weights'] = MLP.get_weights(parameters["Primary_Predictor_params"]['layers'])

        parameters["Readout_Predictor_params"]['layers'] = MLP.get_layers([nhidden+1, 2*nhidden+1, nreadout+1])
        parameters["Readout_Predictor_params"]['beta'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Readout_Predictor_params"]['beta'][0] = 1.0
        parameters["Readout_Predictor_params"]['learning_rate'] = parameters['readout_learning_rate']
        parameters["Readout_Predictor_params"]['momentum'] = parameters['momentum']
        parameters["Readout_Predictor_params"]['mse'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Readout_Predictor_params"]['weights'] = MLP.get_weights(parameters["Readout_Predictor_params"]['layers'])
        parameters["Primary_Predictor_params"]['polynomial'] = polynomial
        parameters["Readout_Predictor_params"]['polynomial'] = polynomial
        parameters['autoencoder'] = autoencoder
        parameters['use_global_backprop'] = use_global_backprop
        parameters["complex_context_in_second_layer"] = complex_context_in_second_layer
        parameters["complex"] = complex_unit

    def min_max_normalize(self, a, min_a, max_a):
        """
        Simple min-max normalization
        :param a: array to be normalized
        :param min_a: recorded min values
        :param max_a: recorded max values
        :return:
        """
        if self.primary_learning_rate[0] != 0:
            min_a[:] = np.minimum(a, min_a) + 0.000001
            max_a[:] = np.maximum(a, max_a) - 0.000001
        return np.divide(a-min_a, max_a-min_a)

    def execute0(self):
        """
        Make predictions based on available information and publish them.

        """
        # Prepare the input vector
        if self.complex and self.complex_context_in_second_layer:
            self.layers_internal_prediction[0]['activation'][:-1] = self.inputs_t
            self.layers_internal_prediction[1]['activation'][-(self.ncontexts+1):-1] = self.contexts_t_1
        else:
            self.layers_internal_prediction[0]['activation'][:-1] = np.concatenate((self.inputs_t, self.contexts_t_1))

        # Make internal prediction
        self.MLP_internal_prediction.mlp_forward()
        # Forward internal representations to the readout predictor
        self.layers_readout[0]['activation'][:-1] = self.layers_internal_prediction[self.output_layer]['activation'][:-1]
        # Make readout prediction
        self.MLP_readout.mlp_forward()
        # Update the output
        if self.normalize:
            self.output_block[:] = self.min_max_normalize((self.layers_internal_prediction[self.output_layer]['activation'][:self.output_length]).reshape(self.output_block.shape), self.output_min, self.output_max)
        else:
            self.output_block[:] = (self.layers_internal_prediction[self.output_layer]['activation'][:self.output_length]).reshape(self.output_block.shape)

        # Publish predictions (mainly for debugging and inspection)
        i = 0
        s = len(self.signal_blocks)
        for (idx, (block, delta, pred_block1, pred_block2)) in enumerate(self.signal_blocks):
            l = np.prod(pred_block1.shape)
            pred_block1[:] = np.clip((self.layers_internal_prediction[-1]['activation'][i:i+l]).reshape(pred_block1.shape), 0, 1)
            pred_block_local = self.internal_buffers[idx][0]
            pred_block_local[:] = np.clip((self.layers_internal_prediction[-1]['activation'][i:i+l]).reshape(pred_block_local.shape), 0, 1)
            if self.predict_2_steps:
                pred_block2[:] = np.clip((self.layers_internal_prediction[-1]['activation'][i+l*s:i+l*s+l]).reshape(pred_block1.shape), 0, 1)
            i += l
        i = 0
        for (block, delta, readout_block) in self.readout_blocks:
            l = np.prod(readout_block.shape)
            readout_block[:] = (self.layers_readout[-1]['activation'][i:i+l]).reshape(readout_block.shape)
            i += l
        # Since predicting two steps into the future, learning can only be performed on previous
        # activations, hence current activations are pushed into the FIFO
        if (self.use_global_backprop or self.predict_2_steps) and self.primary_learning_rate[0] != 0:
            self.push_activation()

    def execute1(self):
        """
        Collect new signals, perform association, prepare for additional learning
        """
        # Collect the current signal
        curent_i = 0
        j = 0
        for (idx, (input_block, delta, pred_block1, pred_block2)) in enumerate(self.signal_blocks):
            if self.flags[0] != 12:
                block = input_block
            else:
                block = pred_block1
            l = np.prod(block.shape)
            forward_i = 0
            # Calculate signal features
            past_block = self.internal_buffers[idx][1]
            self.inputs_t[curent_i:curent_i + l] = block.flatten()
            curent_i += l
            if self.use_derivative:
                self.derivative_blocks[idx][:] = 0.5*(block - past_block)+0.5
                self.inputs_t[curent_i:curent_i+l] = self.derivative_blocks[idx].flatten()
                curent_i += l
            if self.use_integral:
                self.integral_blocks[idx][:] = self.tau[0] * self.integral_blocks[idx] + (1 - self.tau[0]) * block
                self.inputs_t[curent_i:curent_i+l] = self.integral_blocks[idx].flatten()
                curent_i += l
            if self.use_error:
                pred_block_local = self.internal_buffers[idx][0]
                self.error_blocks[idx][:] = 0.5*(block - pred_block_local)+0.5
                self.inputs_t[curent_i:curent_i+l] = self.error_blocks[idx].flatten()
                curent_i += l
            if self.use_t_2_block:
                self.inputs_t[curent_i:curent_i+l] = past_block.flatten()
                curent_i += l
            past_block[:] = block
            # Collect the current signal as supervising signal, unless running as an autoencoder
            if not self.autoencoder:
                self.actual_signal_t[j:j+l] = block.flatten()
                j += l

        # Predictive associative training
        if self.primary_learning_rate[0] != 0:
            # pop the previous activaitons from the FIFO
            if self.predict_2_steps:
                if self.use_global_backprop:
                    self.get_one_step_behind_activation()
                else:
                    self.pop_activation()
                training_signal = np.concatenate((self.actual_signal_t_1, self.actual_signal_t))
            else:
                training_signal = self.actual_signal_t
            # Calculate the prediction error
            error_primary = training_signal - self.layers_internal_prediction[-1]['activation'][:np.prod(training_signal.shape)]
            self.actual_signal_t_1[:] = self.actual_signal_t
            # Make the association
            self.MLP_internal_prediction.train2(error=error_primary)

            if self.use_global_backprop:
                i = 0
                # Extract the deltas for input blocks:
                for (block, delta, pred_block1, pred_block2) in self.signal_blocks:
                    l = np.prod(block.shape)
                    delta[:] = self.layers_internal_prediction[0]['delta'][i:i+l].reshape(delta.shape)
                    i += l * self.input_blocks_skip

                if self.complex and self.complex_context_in_second_layer:
                    base_idx = self.layers_internal_prediction[1]['delta'].shape[0]-self.ncontexts-1
                    for (idx, (block, delta, factor)) in enumerate(self.context_blocks):
                        if factor[0] > 0:
                            l = np.prod(block.shape)
                            delta[:] = self.layers_internal_prediction[1]['delta'][base_idx + idx * l:base_idx + (idx+1) * l].reshape(delta.shape)
                else:
                    for (block, delta, factor) in self.context_blocks:
                        if factor[0] > 0:
                            l = np.prod(block.shape)
                            delta[:] = self.layers_internal_prediction[0]['delta'][i:i + l].reshape(delta.shape)
                            i += l

        # Readout training
        if self.readout_learning_rate[0] != 0:
            # Collect the curent readout training signal
            i = 0
            for (block, delta, pblock) in self.readout_blocks:
                l = np.prod(block.shape)
                self.readout_training_signal[i:i + l] = block.flatten()
                i += l
            error_readout = self.readout_training_signal - self.layers_readout[-1]['activation'][:-1]
            self.MLP_readout.train2(error=error_readout)
            if self.backpropagate_readout_error and self.primary_learning_rate[0] != 0:
                # If true the error generated by the readout will be propagated into the predictive encoder
                self.layers_internal_prediction[self.output_layer]['delta'][:-1] = self.layers_readout[0]['delta'][:-1]
                for l in range(self.output_layer-1, -1, -1):
                    self.MLP_internal_prediction.mlp_backpropagate_layer(layer=l)
                for l in range(self.output_layer-1, -1, -1):
                    self.MLP_internal_prediction.calculate_weight_update_layer(layer=l)
                    self.MLP_internal_prediction.update_weights_layer(layer=l)

        if self.autoencoder:
            j = 0
            for (idx, (block, delta, pred_block1, pred_block2)) in enumerate(self.signal_blocks):
                l = np.prod(block.shape)
                self.actual_signal_t[j:j+l] = block.flatten()
                j += l

    def execute2(self):
        """
        Gather the context signal in preparation for the next step.
        Some context blocks get their information from units lateral to this one,
        others get their information from units above this one.
        The "factor" is either the context_factor_lateral or context_factor_feedback
        """
        if self.primary_learning_rate[0] != 0 and self.use_global_backprop:
            self.pop_activation()
            self.avg_delta *= 0
            for delta in self.delta_blocks:
                self.avg_delta += delta
            self.avg_delta /= len(self.delta_blocks)
            self.layers_internal_prediction[self.output_layer]['delta'][:-1] = self.avg_delta.flatten()
            for l in range(self.output_layer-1, -1, -1):
                self.MLP_internal_prediction.mlp_backpropagate_layer(layer=l)
            for l in range(self.output_layer-1, -1, -1):
                self.MLP_internal_prediction.calculate_weight_update_layer(layer=l)
                self.MLP_internal_prediction.update_weights_layer(layer=l)
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

    def push_activation(self):
        """
        Save the current network activations to a FIFO queue
        :return:
        """
        layers_copy = copy.deepcopy(self.layers_internal_prediction)
        self.activations_buffer.append(layers_copy)

    def pop_activation(self):
        """
        Pop the first available activations from the FIFO
        :return:
        """
        layers = self.activations_buffer.pop(0)
        for layer in range(len(layers)):
            self.layers_internal_prediction[layer]['activation'][:] = layers[layer]['activation']
            self.layers_internal_prediction[layer]['error'][:] = layers[layer]['error']
            self.layers_internal_prediction[layer]['delta'][:] = layers[layer]['delta']

    def get_one_step_behind_activation(self):
        """
        Pop the first available activations from the FIFO
        :return:
        """
        layers = self.activations_buffer[-2]
        for layer in range(len(layers)):
            self.layers_internal_prediction[layer]['activation'][:] = layers[layer]['activation']
            self.layers_internal_prediction[layer]['error'][:] = layers[layer]['error']
            self.layers_internal_prediction[layer]['delta'][:] = layers[layer]['delta']
