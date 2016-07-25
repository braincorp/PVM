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
import PVM_framework.PVM_Create as PVM_Create
import PVM_framework.MLP as MLP
import copy


class ExecutionUnit(AbstractExecutionUnit.ExecutionUnit):
    """
    Unit architecture:

    [internal pred]  +   [residual prediction]
        / \               \          /
       /   \               [OUTPUT]   <- compressed residual error
      /     \             /         \
     [CONTEXT]           [PAST SIGNAL]

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

    UNSUPERVISED_SIGNAL_INPUTS = set(["block: Raw signal block t-1",
                                      "past_block: Raw signal block t-2"])

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

    def __init__(self, parameters):
        # Mapping SharedArray items through .view(np.ndarray) improved the access time
        self.signal_blocks = self.open_views_into_shmem(parameters, "signal_blocks", len(ExecutionUnit.SIGNAL_BLOCK_CONTENTS))
        self.readout_blocks = self.open_views_into_shmem(parameters, "readout_blocks", len(ExecutionUnit.SUPERVISED_TASK_OUTPUTS))
        self.context_blocks = self.open_views_into_shmem(parameters, "context_blocks", len(ExecutionUnit.UNSUPERVISED_CONTEXT_INPUTS))
        self.internal_buffers = self.open_views_into_shmem(parameters, "internal_buffers", 3)
        self.output_block = parameters['output_block'].view(np.ndarray)  # Will appear in other places as either input or context
        self.output_min = parameters['output_min']
        self.output_max = parameters['output_max']

        # The perceptron
        self.MLP_internal_prediction = MLP.MLP(parameters["Primary_Predictor_params"])
        self.MLP_residual_prediction = MLP.MLP(parameters["Residual_Predictor_params"])
        self.MLP_readout = MLP.MLP(parameters["Readout_Predictor_params"])  # This is a "task" supervised MLP (e.g., tracker)
        self.primary_learning_rate = parameters['primary_learning_rate']
        self.readout_learning_rate = parameters['readout_learning_rate']
        self.layers_internal_prediction = parameters["Primary_Predictor_params"]['layers']
        self.layers_residual_prediction = parameters["Residual_Predictor_params"]['layers']
        self.layers_readout = parameters["Readout_Predictor_params"]['layers']  # These are the "task" supervised MLP layers
        self.tau = parameters['tau']  # Tau is the integration constant for the signal integral

        # Input buffers
        self.ninputs = 0
        self.npredictions = 0
        self.npinputs = 0
        self.ncontexts = 0
        for (block, delta, pred_block1, pred_block2) in self.signal_blocks:
            self.ninputs += len(ExecutionUnit.UNSUPERVISED_SIGNAL_INPUTS) * np.prod(block.shape)
            self.npredictions += np.prod(block.shape)
        for (teaching_block, delta, readout_block) in self.readout_blocks:
            self.npinputs += np.prod(teaching_block.shape)
        self.inputs_t = np.zeros((self.ninputs,))
        self.actual_signal_t = np.zeros((self.npredictions,))
        self.readout_training_signal = np.zeros((self.npinputs,))

        self.inputs_t_1 = np.zeros((self.ninputs,))
        self.actual_signal_t_1 = np.zeros((self.npredictions,))
        self.pinputs_t_1 = np.zeros((self.npinputs,))

        # Context buffers
        for (block, delta, factor) in self.context_blocks:
            self.ncontexts += np.prod(block.shape)
        self.contexts_t_1 = np.zeros((self.ncontexts,))
        self.output_layer = len(self.layers_residual_prediction)-2  # Becuse the "output" of this Unit comes from its hidden layer
        self.output_length = np.prod(self.output_block.shape)
        # Buffer for storing activations
        self.activations_buffer = []
        # additional flags
        if 'complex' in parameters.keys() and parameters['complex']:
            self.complex = True
        else:
            self.complex = False
        self.push_activation()

    @staticmethod
    def upgrade_to_ver_1(parameters):
        parameters['internal_buffers'] = []
        parameters['output_min'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])
        parameters['output_max'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])
        for (block, delta, pred_block, pred_block2) in parameters['signal_blocks']:
            block01 = SharedArray.SharedNumpyArray_like(block)
            block02 = SharedArray.SharedNumpyArray_like(block)
            block03 = SharedArray.SharedNumpyArray_like(block)
            parameters['internal_buffers'].append((block01, block02, block03))

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

        nhidden = np.prod(parameters['output_block'].shape)
        parameters['output_min'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])
        parameters['output_max'] = SharedArray.SharedNumpyArray_like(parameters['output_block'])
        ninputs = 0
        noutputs = 0
        ncontext = 0
        # Any additional memory buffers needed in the operation of the unit
        parameters['internal_buffers'] = []
        for (block, delta, pred_block, pred_block2) in parameters['signal_blocks']:
            block01 = SharedArray.SharedNumpyArray_like(block)
            block02 = SharedArray.SharedNumpyArray_like(block)
            block03 = SharedArray.SharedNumpyArray_like(block)
            parameters['internal_buffers'].append((block01, block02, block03))

        for (block, delta, pred_block, pred_block2) in parameters['signal_blocks']:
            ninputs += np.prod(block.shape) * len(ExecutionUnit.UNSUPERVISED_SIGNAL_INPUTS)
        for (block, delta, factor) in parameters['context_blocks']:
            ncontext += np.prod(block.shape)
        for (block, delta, pred_block, pred_block2) in parameters['signal_blocks']:
            noutputs += 2 * np.prod(block.shape)  # to predict two steps into the future
        nadditional = 0
        for (block, delta, pblock) in parameters['readout_blocks']:
            nadditional += np.prod(block.shape)
        parameters["Primary_Predictor_params"] = {}
        parameters["Residual_Predictor_params"] = {}
        parameters["Readout_Predictor_params"] = {}
        if complex_unit:  # 4 layer perceptron
            parameters["Primary_Predictor_params"]['layers'] = MLP.get_layers([ncontext+1, 3*nhidden+1, 2*nhidden+1, noutputs+1])
            parameters["Residual_Predictor_params"]['layers'] = MLP.get_layers([ninputs+1, 2*nhidden+1, nhidden+1, noutputs+1])
            parameters["complex"] = True
        else:  # 3 layer perceptron Simple MLP Unit (not complex unit)
            parameters["Primary_Predictor_params"]['layers'] = MLP.get_layers([ncontext+1, 2*nhidden+1, noutputs+1])
            parameters["Residual_Predictor_params"]['layers'] = MLP.get_layers([ninputs+2*nhidden+1, nhidden+1, noutputs+1])

        parameters["Primary_Predictor_params"]['beta'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Primary_Predictor_params"]['beta'][0] = 1.0
        parameters["Primary_Predictor_params"]['learning_rate'] = parameters['primary_learning_rate']
        parameters["Primary_Predictor_params"]['momentum'] = parameters['momentum']
        parameters["Primary_Predictor_params"]['mse'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Primary_Predictor_params"]['weights'] = MLP.get_weights(parameters["Primary_Predictor_params"]['layers'])

        parameters["Residual_Predictor_params"]['beta'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Residual_Predictor_params"]['beta'][0] = 1.0
        parameters["Residual_Predictor_params"]['learning_rate'] = parameters['primary_learning_rate']
        parameters["Residual_Predictor_params"]['momentum'] = parameters['momentum']
        parameters["Residual_Predictor_params"]['mse'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Residual_Predictor_params"]['weights'] = MLP.get_weights(parameters["Residual_Predictor_params"]['layers'])

        parameters["Readout_Predictor_params"]['layers'] = MLP.get_layers([nhidden+1, 2*nhidden+1, nadditional+1])
        parameters["Readout_Predictor_params"]['beta'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Readout_Predictor_params"]['beta'][0] = 1.0
        parameters["Readout_Predictor_params"]['learning_rate'] = parameters['readout_learning_rate']
        parameters["Readout_Predictor_params"]['momentum'] = parameters['momentum']
        parameters["Readout_Predictor_params"]['mse'] = SharedArray.SharedNumpyArray((1,), np.float)
        parameters["Readout_Predictor_params"]['weights'] = MLP.get_weights(parameters["Readout_Predictor_params"]['layers'])
        parameters["Primary_Predictor_params"]['polynomial'] = polynomial
        parameters["Residual_Predictor_params"]['polynomial'] = polynomial
        parameters["Readout_Predictor_params"]['polynomial'] = polynomial
        parameters['autoencoder'] = autoencoder

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
        self.layers_internal_prediction[0]['activation'][:-1] = self.contexts_t_1
        # Make internal prediction
        self.MLP_internal_prediction.mlp_forward()
        # Make residual prediction
        self.layers_residual_prediction[0]['activation'][:-1] = np.concatenate((self.inputs_t, self.layers_internal_prediction[1]["activation"][:-1]))
        self.MLP_residual_prediction.mlp_forward()
        # Forward internal representations to the readout predictor
        self.layers_readout[0]['activation'][:-1] = self.layers_residual_prediction[self.output_layer]['activation'][:-1]
        # Make readout prediction
        self.MLP_readout.mlp_forward()
        # Update the output
        self.output_block[:] = self.min_max_normalize((self.layers_residual_prediction[self.output_layer]['activation'][:self.output_length]).reshape(self.output_block.shape), self.output_min, self.output_max)
        # Publish predictions (mainly for debugging and inspection)
        i = 0
        s = len(self.signal_blocks)
        for (idx, (block, delta, pred_block1, pred_block2)) in enumerate(self.signal_blocks):
            l = np.prod(pred_block1.shape)
            pred_block1[:] = np.clip((self.layers_internal_prediction[-1]['activation'][i:i+l] +
                                     2 * (self.layers_residual_prediction[-1]['activation'][i:i+l]-0.5)).reshape(pred_block1.shape), 0, 1)
            pred_block_local = self.internal_buffers[idx][0]
            pred_block_local[:] = np.clip((self.layers_internal_prediction[-1]['activation'][i:i+l] +
                                           2*(self.layers_residual_prediction[-1]['activation'][i:i+l]-0.5)).reshape(pred_block_local.shape), 0, 1)
            pred_block2[:] = np.clip((self.layers_internal_prediction[-1]['activation'][s*l+i:s*l+i+l] +
                                     2 * (self.layers_residual_prediction[-1]['activation'][s*l+i:s*l+i+l]-0.5)).reshape(pred_block1.shape), 0, 1)
            i += l
        i = 0
        for (block, delta, readout_block) in self.readout_blocks:
            l = np.prod(readout_block.shape)
            readout_block[:] = (self.layers_readout[-1]['activation'][i:i+l]).reshape(readout_block.shape)
            i += l
        # Since predicting two steps into the future, learning can only be performed on previous
        # activations, hence current activations are pushed into the FIFO
        if self.primary_learning_rate[0] != 0:
            self.push_activation()

    def execute1(self):
        """
        Collect new signals, perform association, prepare for additional learning
        """
        # Collect the current signal
        i = 0
        j = 0
        for (idx, (block, delta, pred_block1, pred_block2)) in enumerate(self.signal_blocks):
            l = np.prod(block.shape)
            # Calculate signal features
            past_block = self.internal_buffers[idx][1]
            self.inputs_t[i:i + l] = block.flatten()
            self.inputs_t[i + l:i + 2 * l] = past_block.flatten()
            past_block[:] = block
            i += 2*l
            # Collect the current signal as supervising signal
            self.actual_signal_t[j:j+l] = block.flatten()
            j += l
        # Predictive associative training
        if self.primary_learning_rate[0] != 0:
            # pop the previous activaitons from the FIFO
            self.pop_activation()
            # Create the expected output block
            training_signal = np.concatenate((self.actual_signal_t_1, self.actual_signal_t))
            # Calculate the prediction error
            error_primary = training_signal - self.layers_internal_prediction[-1]['activation'][:np.prod(training_signal.shape)]
            error_residual = (error_primary + 1) * 0.5 - self.layers_residual_prediction[-1]['activation'][:np.prod(training_signal.shape)]
            self.actual_signal_t_1[:] = self.actual_signal_t
            # Make the association
            self.MLP_internal_prediction.train2(error=error_primary)
            self.MLP_residual_prediction.train2(error=error_residual)
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

    def push_activation(self):
        """
        Save the current network activations to a FIFO queue
        :return:
        """
        layers_copy = copy.deepcopy(self.layers_internal_prediction)
        layers_res_copy = copy.deepcopy(self.layers_residual_prediction)
        self.activations_buffer.append((layers_copy, layers_res_copy))

    def pop_activation(self):
        """
        Pop the first available activations from the FIFO
        :return:
        """
        (layers, layers_res) = self.activations_buffer.pop(0)
        for layer in range(len(layers)):
            self.layers_internal_prediction[layer]['activation'][:] = layers[layer]['activation']
            self.layers_internal_prediction[layer]['error'][:] = layers[layer]['error']
            self.layers_internal_prediction[layer]['delta'][:] = layers[layer]['delta']
        for layer in range(len(layers_res)):
            self.layers_residual_prediction[layer]['activation'][:] = layers_res[layer]['activation']
            self.layers_residual_prediction[layer]['error'][:] = layers_res[layer]['error']
            self.layers_residual_prediction[layer]['delta'][:] = layers_res[layer]['delta']
