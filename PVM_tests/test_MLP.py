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

import PVM_framework.MLP as MLP_module
import numpy as np


def test_perceptron01():
    X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    Y = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
    parameters = {}
    parameters['layers'] = MLP_module.get_layers([3, 5, 3])
    np.random.seed(seed=509)
    parameters['weights'] = MLP_module.get_weights(parameters['layers'])
    parameters['beta'] = np.array([1.0])
    parameters['learning_rate'] = np.array([0.1])
    parameters['momentum'] = np.array([0.5])
    parameters['mse'] = np.array([0.0])
    M = MLP_module.MLP(parameters)
    for n in xrange(30000):
        i = np.random.randint(low=0, high=4)
        M.train(X[i], Y[i])
    for i in range(4):
        O = M.evaluate(X[i])
        assert(np.allclose(Y[i], O, atol=0.03))


def test_perceptron02():
    X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    Y = np.array([[1], [0], [1], [0]])
    parameters = {}
    parameters['layers'] = MLP_module.get_layers([3, 3, 2])
    np.random.seed(seed=509)
    parameters['weights'] = MLP_module.get_weights(parameters['layers'])
    parameters['beta'] = np.array([1.0])
    parameters['learning_rate'] = np.array([0.25])
    parameters['momentum'] = np.array([0.5])
    parameters['mse'] = np.array([0.0])
    M = MLP_module.MLP(parameters)
    for n in xrange(30000):
        i = np.random.randint(low=0, high=4)
        M.train(X[i], Y[i])
    for i in range(4):
        O = M.evaluate(X[i])
        assert(np.allclose(Y[i], O, atol=0.03))


def test_perceptron_two_layers():
    X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    Y = np.array([[1], [1], [1], [0]])
    parameters = {}
    parameters['layers'] = MLP_module.get_layers([3, 2])
    np.random.seed(seed=509)
    parameters['weights'] = MLP_module.get_weights(parameters['layers'])
    parameters['beta'] = np.array([1.0])
    parameters['learning_rate'] = np.array([0.25])
    parameters['momentum'] = np.array([0.5])
    parameters['mse'] = np.array([0.0])
    M = MLP_module.MLP(parameters)
    for n in xrange(30000):
        i = np.random.randint(low=0, high=4)
        M.train(X[i], Y[i])
    for i in range(4):
        O = M.evaluate(X[i])
        assert(np.allclose(Y[i], O, atol=0.03))


def test_MLPN_eval00():
    """
    Simple test checking whether the MLP evaluates properly with zero weights
    """
    x = np.array([1.0, 0.0])
    parameters = {}
    parameters['layers'] = MLP_module.get_layers([3, 5, 5])
    parameters['beta'] = np.array([1.0])
    parameters['learning_rate'] = np.array([1.0])
    parameters['momentum'] = np.array([0.0])
    parameters['mse'] = np.array([0.0])
    parameters['weights'] = [np.array([[0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]]).astype(np.float)*100,
                             np.array([[0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]]).astype(np.float)*100
                             ]
    M = MLP_module.MLP(parameters)
    out = M.evaluate(x)
    assert np.allclose(out, np.array([0.5, 0.5, 0.5, 0.5]))
    hidden = M.get_activation(1)
    assert np.allclose(hidden, np.array([0.5, 0.5, 0.5, 0.5]))


def test_MLPN_eval01():
    """
    Simple test checking whether the MLP evaluates properly on a specific case
    """
    x3 = np.array([0.0, 1.0])
    x4 = np.array([1.0, 0.0])
    parameters = {}
    parameters['layers'] = MLP_module.get_layers([3, 6, 5])
    parameters['beta'] = np.array([1.0])
    parameters['learning_rate'] = np.array([1.0])
    parameters['momentum'] = np.array([0.0])
    parameters['mse'] = np.array([0.0])
    parameters['polynomial'] = True
    parameters['weights'] = [np.array([[1, -1, 1, -1],
                                       [1, -1, -1, 1],
                                       [0, 0, 0, 0]]).astype(np.float)*100,
                             np.array([[1, 0, 0, 0],
                                       [0, 2, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, -1, 2],
                                       [0, 0, 0, 0],
                                       [0, -1, 0, -1],
                                       ]).astype(np.float)*100
                             ]
    M = MLP_module.MLP(parameters)
    out = M.evaluate(x4)
    assert (out[0] > 0.99)
    assert (out[1] < 0.01)
    assert (out[2] > 0.99)
    assert (out[3] < 0.01)
    hidden = M.get_activation(1)
    assert (hidden[0] > 0.99)
    assert (hidden[1] < 0.01)
    assert (hidden[2] > 0.99)
    assert (hidden[3] < 0.01)
    out = M.evaluate(x3)
    assert (out[0] > 0.99)
    assert (out[1] < 0.01)
    assert (out[2] < 0.01)
    assert (out[3] > 0.99)
    hidden = M.get_activation(1)
    assert (hidden[0] > 0.99)
    assert (hidden[1] < 0.01)
    assert (hidden[2] < 0.01)
    assert (hidden[3] > 0.99)


def test_MLPN_xor():
    """
    Simple test checking whether a 3 layer MLP is able to learn XOR
    """
    x1 = np.array([0.0, 0.0])
    x2 = np.array([1.0, 1.0])
    x3 = np.array([0.0, 1.0])
    x4 = np.array([1.0, 0.0])
    parameters = {}
    parameters['layers'] = MLP_module.get_layers([3, 15, 3])
    np.random.seed(seed=509)
    parameters['weights'] = MLP_module.get_weights(parameters['layers'])
    parameters['beta'] = np.array([1.0])
    parameters['learning_rate'] = np.array([0.1])
    parameters['momentum'] = np.array([0.5])
    parameters['mse'] = np.array([0.0])
    M = MLP_module.MLP(parameters)
    for n in xrange(30000):
        M.train(x1, np.array([1.0, 0.0]))
        M.train(x2, np.array([1.0, 0.0]))
        M.train(x3, np.array([0.0, 1.0]))
        M.train(x4, np.array([0.0, 1.0]))
    o1 = M.evaluate(x1)
    o2 = M.evaluate(x2)
    o3 = M.evaluate(x3)
    o4 = M.evaluate(x4)
    assert o1[0] > 0.8
    assert o1[1] < 0.2
    assert o2[0] > 0.8
    assert o2[1] < 0.2
    assert o3[0] < 0.2
    assert o3[1] > 0.8
    assert o4[0] < 0.2
    assert o4[1] > 0.8


def test_MLPN_xor_poly():
    """
    Simple test checking whether a 3 layer MLP is able to learn XOR
    """
    x1 = np.array([0.0, 0.0])
    x2 = np.array([1.0, 1.0])
    x3 = np.array([0.0, 1.0])
    x4 = np.array([1.0, 0.0])
    parameters = {}
    parameters['layers'] = MLP_module.get_layers([3, 15, 3])
    np.random.seed(seed=509)
    parameters['weights'] = MLP_module.get_weights(parameters['layers'])
    parameters['beta'] = np.array([1.0])
    parameters['learning_rate'] = np.array([0.1])
    parameters['momentum'] = np.array([0.5])
    parameters['mse'] = np.array([0.0])
    parameters['polynomial'] = True
    M = MLP_module.MLP(parameters)
    for n in xrange(30000):
        M.train(x1, np.array([1.0, 0.0]))
        M.train(x2, np.array([1.0, 0.0]))
        M.train(x3, np.array([0.0, 1.0]))
        M.train(x4, np.array([0.0, 1.0]))
    o1 = M.evaluate(x1)
    o2 = M.evaluate(x2)
    o3 = M.evaluate(x3)
    o4 = M.evaluate(x4)
    assert o1[0] > 0.8
    assert o1[1] < 0.2
    assert o2[0] > 0.8
    assert o2[1] < 0.2
    assert o3[0] < 0.2
    assert o3[1] > 0.8
    assert o4[0] < 0.2
    assert o4[1] > 0.8


if __name__ == '__main__':
    test_MLPN_eval00()
    test_MLPN_eval01()
    test_MLPN_xor()
    test_MLPN_xor_poly()
    test_perceptron_two_layers()
