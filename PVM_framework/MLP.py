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
import abc
import PVM_framework.fast_routines as fast_routines
import PVM_framework.SharedArray as SharedArray
import time

try:
    @profile
    def test():
        pass
except:
    def profile(fun):
        return fun


def random_ortho(n):
    """ Generates a random orthogonal matrix n x n

    :param n: dimesion
    :type n: int

    """
    A = np.mat(np.random.random((n, n)))
    Q, R = np.linalg.qr(A)
    return Q


def random_with_singular_values(m, n, singular_values):
    """
    Generates a random matrix with given singular values

    :param m: first dimension of the array
    :type m: int
    :param n: second dimension of the array
    :type n: int
    :param singular_values: value of the singuar values of the generated matrix
    :type singular_values: float
    :return: matrix
    :rtype: np.ndarray
    """
    Q = random_ortho(m)
    singular_values=np.array(singular_values)
    svs = singular_values.shape[0]
    if svs < max(n, m):
        singular_values = np.concatenate((singular_values, np.array([0] * (max(n, m)-svs))))
    D = np.diag(singular_values)
    V = random_ortho(n)
    M = Q*D[:m, :n]*V
    return np.array(M)


def optimal_weight_initialization(m, n, singular_value=1.1):
    """ Optimal weight matrix initialization """
    return random_with_singular_values(m, n, [singular_value] * min(m, n))


class MLPPrototype(object):
    """
    MLPPrototype is an abstract class to be extended by particular implementations of
    the multi layer perceptron.
    """
    def __init__(self, state):
        pass

    @abc.abstractmethod
    def train(self, inputs, targets):
        """
        Pass and train one vector consisting of:

          * inputs - 1 dimensional numpy array
          * targets - 1 dimensional numpy array

        returns the value of the perceptron as evaluated on inputs
        """

    @abc.abstractmethod
    def evaluate(self, inputs):
        """
        Returns the evaluation of a perceptron on inputs (1-d numpy array)
        """
    @abc.abstractmethod
    def copy_weights(self):
        """
        Assures that the weight matrix is stored back in the state dictionary

        """
    @abc.abstractmethod
    def get_hidden(self, layer):
        """
        Returns the activations of the hidden layer
        """

    @abc.abstractmethod
    def get_deltas(self, layer):
        """
        Returns the deltas computed by the backprop algorithm
        in the previous call of train.
        """

    @abc.abstractmethod
    def copy_state(self):
        """
        Copies all the state values into the state that appear
        in the paramaters dictionary. If the implementation already uses
        these arrays, the method can just pass
        """


def view_as_ndarray(object):
    """
    This function takes a nested collection of lists/dictionaries with leafs that are numpy array like objects
    (e.g. shared numpy array) and casts them to a numpy array object which may accelerate the access time.

    :param object: nested collection of dictionaries/lists with leafs that are numpy array like objects.

    .. note::
        If the leafs of the array are shared memory objects, running this method may render them non serializable,
        in other words, they will serialize as regular numpy arrays.
    """
    if type(object) is list:
        retval = []
        for element in object:
            retval.append(view_as_ndarray(element))
        return retval
    if type(object) is dict:
        retval = {}
        for element in object.keys():
            retval[element] = view_as_ndarray(object[element])
        return retval
    if object is None:
        return None
    try:
        retval = object.view(np.ndarray)
        return retval
    except:
        raise Exception("Object neither a dict, nor a list and not an ndarray like")


def get_layers(dims=None):
    """
    Instantiate MLP layers with given dimensions
    :param dims: list of ints
    :return: layer dictinary
    """
    if not dims:
        dims = []
    layers = []
    for d in dims:
        layers.append({'activation': SharedArray.SharedNumpyArray(d, np.float),
                       'error': SharedArray.SharedNumpyArray(d, np.float),
                       'delta': SharedArray.SharedNumpyArray(d, np.float)
                       })
    return layers


def get_weights(layers=None):
    """
    Instantiate MLP weights
    :param layers: layer dictionary
    :return: list of weight matrices
    """
    if not layers:
        layers = []
    mlp_weights = []
    for l in range(len(layers)-1):
        l0 = layers[l]['activation'].shape[0]
        l1 = layers[l+1]['activation'].shape[0] - 1
        mlp_weights.append(initialize_weights(SharedArray.SharedNumpyArray((l0, l1), np.float)))
    return mlp_weights


def initialize_weights(w, method=0):
    """
    Initialize a weight matrix. Two methods are supported:

    method 0 is the classical initialization with uniform random variable around zero with variance inversely
             proportional to the first matrix dimension
    method 1 creates a random matrix with all singular values equal to 1.1.

    :param w: array to be populated with values
    :type w: numpy.ndarray
    :param method: method to be used
    :type method: int
    :return: referece to the populated array
    :rtype: numpy.ndarray
    """
    if method == 1:
        w[:] = optimal_weight_initialization(w.shape[0], w.shape[1])
    elif method == 0:
        pass
        w[:] = (np.random.rand(w.shape[0], w.shape[1])-0.5)*2/np.sqrt(w.shape[0]-1)
    else:
        raise("Unknown initialization method!")
    return w


class MLP(MLPPrototype):
    """
    Multi Layer Perceptron with arbitrary number of layers.

    This sample code implements the MLPPrototype abstract class using pure Python
    and numpy.

    A distinctive feature of this code is that a lot of the data structures containing the state
    of the algorithm (e.g., weight matrices) are supplied externally to the constructor. The constructor then
    uses references to these arrays are internal state objects. This allows this code to be used in a parallel
    system, where state variables can be instantiated as shared memory arrays, and inspected, saved or modified
    externally.


    :param state: dictionary containing predefined fields. The fields describe the network
    structure and provide allocated arrays (numpy arrays) to store the calculated weight values and
    other data.
    :type state: dict

    This design allows the object to have external side effects, particularly side effects in a different process
    if supplied data strucutres are shared.

    The dictionary should contain:

    state['layers'] - list of layers each layer being a dictionary with three keys:

        * activation - a numpy array like object for storing activations (including the bias activation always
            equal to 1, being the last element)
        * error - a numpy array like object of the same shape as activation for storing the error
        * delta - a numpy array like object of the same shape as activation for storing the delta factors


    state['beta'] - numpy array like object containing a scaling factor for the sigmoid
    state['momentum'] - numpy array like object containing a  momentum term for learning
    state['learning_rate'] - numpy array like object containing the learning rate
    state['mse'] - a single element numpy array that will be populated with the calculated mean squared error (MSE)
    state['weights'] - a list of weight matrices with compatible shapes.

    The shapes of weights should be:

        * weights[0] - weights from layer 0 to layer 1. The shape should be A_l0 x (A_l1 - 1) where A_lX stands for
            the shape of activation array in layer X
        * weights[1] - weights from layer 1 to layer 2. The shape should be A_l1 x (A_l2 - 1)
        * weights[2] - weights from layer 2 to layer 3. The shape should be A_l2 x (A_l3 - 1)
        * and so on

    :Example:
    ::

        X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
        Y = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        state = {}
        state['layers'] = [
            {'activation': np.zeros((3,)), 'error': np.zeros((3,)), 'delta': np.zeros((3,))},
            {'activation': np.zeros((5,)), 'error': np.zeros((5,)), 'delta': np.zeros((5,))},
            {'activation': np.zeros((3,)), 'error': np.zeros((3,)), 'delta': np.zeros((3,))},
        ]
        state['weights'] = [
            MLP2.initialize_weights(np.zeros((3, 4)), False),
            MLP2.initialize_weights(np.zeros((5, 2)), False)
        ]
        state['beta'] = np.array([1.0])
        state['learning_rate'] = np.array([0.1])
        state['momentum'] = np.array([0.5])
        state['mse'] = np.array([0.0])
        M = MLP2(state)
        for n in xrange(20000):
            i = np.random.randint(low=0, high=4)
            M.train(X[i], Y[i])
        for i in range(4):
            O = M.evaluate(X[i])
            assert(np.allclose(Y[i], O, atol=0.05))

    """

    def copy_weights(self):
        pass

    def get_hidden(self, layer):
        pass

    def __init__(self, state):
        # Set up network size
        self.layers = view_as_ndarray(state['layers'])
        for layer in self.layers:
            assert 'activation' in layer.keys()
            assert 'error' in layer.keys()
            assert 'delta' in layer.keys()

        self.n_layers = len(self.layers)
        self.beta = state['beta']
        self.momentum = state['momentum'].view(np.ndarray)
        self.learning_rate = state['learning_rate'].view(np.ndarray)
        # Initialize network
        self.weights = view_as_ndarray(state['weights'])
        self.delta_w = []
        self.__prev_update = []
        for i in range(len(self.weights)):
            assert(self.layers[i]['activation'].shape[0] == self.weights[i].shape[0])
            if i > 0:
                assert(self.layers[i]['activation'].shape[0] >= self.weights[i-1].shape[1]+1)
            self.delta_w.append(np.zeros_like(self.weights[i]))
            self.__prev_update.append(np.zeros_like(self.weights[i]))
        self.mse = state['mse'].view(np.ndarray)
        for layer in self.layers:
            if "no_bias" in state.keys():
                layer['activation'][-1] = 0
            else:
                layer['activation'][-1] = 1
        self.polynomial = False
        if "polynomial" in state:
            self.polynomial = True

    def train(self, inputs, targets, eta=None):
        """
        Train one sample of data

        :param inputs: one dimensional numpy array containing the input vector
        :type inputs: np.ndarray
        :param targets: one dimesinonal numpy array containing the desired output
        :type targets: np.ndarray
        :param eta : override the learning rate
        :type eta: float

        :return: actual activation of the output layer
        :rtype: np.ndarray
        """
        if eta is None:
            eta = self.learning_rate[0]
        self.mlp_forward(inputs)
        outputs = self.layers[-1]['activation'][:-1]
        self.layers[-1]['error'][:-1] = targets - outputs
        if not self.polynomial:
            fast_routines.derivative_dot_c(outputs, self.layers[-1]['error'], self.layers[-1]['delta'])
        else:
            fast_routines.derivative_dot_c_poly(outputs, self.layers[-1]['error'], self.layers[-1]['delta'])
        self.mlp_backpropagate()
        self.calculate_weight_update()
        self.update_weights()
        return outputs.copy()

    def evaluate(self, inputs):
        """
        Evaluate the perceptron on a sample without learning

        :param inputs: one dimensional numpy array of inputs
        :type inputs: numpy.ndarray

        :return: activation of the output layer
        :rtype: np.ndarray
        """
        self.mlp_forward(inputs)
        return self.layers[-1]['activation'][:-1].copy()

    def train2(self, error, eta=None):
        """
        Similar to train only here the user supplies the error which allows for a little bit more
        flexibility (e.g. part of the error be a direct function of the results)

        :param error: one dimensional numpy array of error
        :type error: numpy.ndarray
        :param eta: override the learning rate
        :type eta: float
        :return: numpy.ndarray
        :rtype: numpy.ndarray
        """
        if eta is None:
            eta = self.learning_rate[0]
        self.layers[-1]['error'][:] = 0
        self.layers[-1]['error'][:error.shape[0]] = error
        if not self.polynomial:
            fast_routines.derivative_dot_c(self.layers[-1]['activation'], self.layers[-1]['error'], self.layers[-1]['delta'])
        else:
            fast_routines.derivative_dot_c_poly(self.layers[-1]['activation'], self.layers[-1]['error'], self.layers[-1]['delta'])
        self.mlp_backpropagate()
        self.calculate_weight_update()
        self.update_weights()

    def mlp_forward(self, inputs=None):
        """
        Run the network forward. This method should not be used externally.

        :param inputs: one dimensional numpy array of inputs
        :type inputs: numpy.ndarray
        """
        if inputs is not None:
            self.layers[0]['activation'][:-1] = inputs
        for layer in xrange(self.n_layers-1):
            self.mlp_forward_layer(layer)

    def mlp_forward_layer(self, layer):
        """
        Propagate activations forward from layer to layer + 1

        :param layer: index of the layer to propagate from
        :type layer: int
        :return:
        """
        input = self.layers[layer]['activation']
        weights = self.weights[layer]
        beta = self.beta[0]
        result = self.layers[layer+1]['activation']
        if not self.polynomial:
            fast_routines.dot_sigmoid_c(input, weights, beta, result, 0)
        else:
            fast_routines.dot_sigmoid_c_poly(input, weights, beta, result, 0)

    def mlp_backpropagate_layer(self, layer):
        """
        Propagate error backwards from layer + 1 to layer

        :param layer: index of the layer to propagate the error to
        :type layer: int
        :return:
        """
        deltas = self.layers[layer]['delta']
        upper_deltas = self.layers[layer+1]['delta']
        activation = self.layers[layer]['activation']
        weights = self.weights[layer]
        error = self.layers[layer]['error']
        fast_routines.dot_transpose_c(upper_deltas[:weights.shape[1]], weights, error)
        if not self.polynomial:
            fast_routines.derivative_dot_c(activation, error, deltas)
        else:
            fast_routines.derivative_dot_c_poly(activation, error, deltas)

    def mlp_backpropagate(self):
        """
        Propagate the error backwards from the output to the input layer

        :return:
        """
        for layer in xrange(self.n_layers-2, -1, -1):
            self.mlp_backpropagate_layer(layer)

    def calculate_weight_update_layer(self, layer, eta=None):
        """
        Given the backpropagated error and delta factors, calculate the weight update

        :param layer:
        :return:
        """
        if eta is None:
            eta = self.learning_rate[0]
        inputs = self.layers[layer]['activation']
        deltas = self.layers[layer+1]['delta']
        fast_routines.generalized_outer_c(eta, inputs, deltas[:self.weights[layer].shape[1]], self.momentum[0], self.__prev_update[layer], self.delta_w[layer])

    def calculate_weight_update(self, eta=None):
        """
        Calculate weight updates for all the layers.

        :return
        """
        for layer in xrange(len(self.weights)):
            self.calculate_weight_update_layer(layer, eta=eta)

    def update_weights_layer(self, layer):
        """
        Perform the actual weight update for a layer of weights. Note layer here is the index of the
        input layer for the weight matrix that will be modified.

        :param layer: layer index
        :type layer: int
        :return:
        """
        self.weights[layer] += self.delta_w[layer]
        self.__prev_update[layer][:] = self.delta_w[layer][:]

    def update_weights(self):
        """
        Update weights in all layers.

        :return:
        """
        for layer in xrange(len(self.weights)):
            self.update_weights_layer(layer)

    def get_activation(self, layer=0):
        """
        Get the activation of the layers (updated when self.train or self.evaluate was called)

        :param layer: index of the layer (0 - input, -1 output)
        :type layer: int
        :return: activation of the hidden layer
        :rtype: numpy.ndarray
        """
        return self.layers[layer]['activation'][:-1]

    def get_deltas(self, layer=0):
        """
        Get the deltas calculated by backprop on hidden layers (updated when self.train was called)

        :param layer : index of the  layer
        :type layer: int
        :return: backpropagated error for every unit in the hidden layer
        :rtype: numpy.ndarray
        """
        return self.layers[layer]['delta']

    def copy_state(self):
        """
        An empty method, since all of the state is actually kept in externally supplied data
        structures.

        :return: No return value
        """
        pass


if __name__ == "__main__":
    # Timing with and without flushing denormals
    X = np.random.rand(100, 100)
    Y = np.random.rand(100, 100)
    state = {}
    state['layers'] = [
        {'activation': np.zeros((101,)), 'error': np.zeros((101,)), 'delta': np.zeros((101,))},
        {'activation': np.zeros((51,)), 'error': np.zeros((51,)), 'delta': np.zeros((51,))},
        {'activation': np.zeros((101,)), 'error': np.zeros((101,)), 'delta': np.zeros((101,))},
    ]
    state['weights'] = [
        MLP.initialize_weights(np.zeros((101, 50)), False),
        MLP.initialize_weights(np.zeros((51, 100)), False)
    ]
    state['beta'] = np.array([1.0])
    state['learning_rate'] = np.array([0.1])
    state['momentum'] = np.array([1.0])
    state['mse'] = np.array([0.0])
    M = MLP(state)
    t_start = time.clock()
    try:
        import LowLevelCPUControlx86
        LowLevelCPUControlx86.set_flush_denormals()
    except:
        # Unable to set the flags
        print "Setting flush denormals CPU flag not available"
    for n in xrange(100000):
        i = n % 100
        M.train(X[i], Y[i])
    t_exec = time.clock()-t_start
    print "Execution time %f" % t_exec
