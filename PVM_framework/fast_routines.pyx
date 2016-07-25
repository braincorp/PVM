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

import cython
import numpy as np
cimport numpy as np
from cpython.buffer cimport PyBUF_SIMPLE
from cpython.buffer cimport Py_buffer
from cpython.buffer cimport PyObject_GetBuffer
from cpython.buffer cimport PyBuffer_Release
import os
from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
np.import_array()


cdef extern from "Accelerated.h":
    cdef void dot_transpose(double* mult, double* vector, int vect_shape_0, double* matrix, int mat_shape0, int mat_shape1, double* result)
    cdef void dot_transpose_simple(double* vector, int vect_shape_0, double* matrix, int mat_shape0, int mat_shape1, double* result)
    cdef void derivative_dot(double* vector, double* vector2, int vect_shape_0, double* result)
    cdef void derivative_dot_poly(double* vector, double* vector2, int vect_shape_0, double* result)
    cdef void generalized_outer(double alpha, double * vector1, int vect1_shape, double * vector2, int vect2_shape, double beta, double* matrix, double* result)
    cdef void dot_sigmoid(double* vector, double* matrix, int mat_shape0, int mat_shape1, double beta, double * result, int append_bias)
    cdef void dot_sigmoid_poly(double* vector, double* matrix, int mat_shape0, int mat_shape1, double beta, double * result, int append_bias)
    cdef void dot_add(double* vector, double* matrix, int mat_shape0, int mat_shape1, double * result, int append_bias)
    cdef void sigmoid_poly(double* result, int shape, double beta)
    cdef void sigmoid(double* result, int shape, double beta)


@cython.boundscheck(False)
def derivative_dot_cython(np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=1, mode='c'] vector2 not None, np.ndarray[double, ndim=1, mode='c'] result not None):
    result[:]=vector * (1.0 - vector) * vector2


@cython.boundscheck(False)
def derivative_dot_c(np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=1, mode='c'] vector2 not None, np.ndarray[double, ndim=1, mode='c'] result not None):
    cdef int vs0
    vs0=vector.shape[0]
    derivative_dot(& vector[0], & vector2[0], vs0, & result[0])


def derivative_dot_c_poly(np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=1, mode='c'] vector2 not None, np.ndarray[double, ndim=1, mode='c'] result not None):
    cdef int vs0
    vs0=vector.shape[0]
    derivative_dot_poly(& vector[0], & vector2[0], vs0, & result[0])


@cython.boundscheck(False)
def dot_transpose_mul_cython(np.ndarray[double, ndim=1, mode='c'] mult not None, np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=2, mode='c'] matrix not None, np.ndarray[double, ndim=1, mode='c'] result not None):
    result[:]=mult*np.dot(vector, matrix.T)


@cython.boundscheck(False)
def dot_transpose_cython(np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=2, mode='c'] matrix not None, np.ndarray[double, ndim=1, mode='c'] result not None):
    result[:]=np.dot(vector, matrix.T)


@cython.boundscheck(False)
def dot_transpose_c(np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=2, mode='c'] matrix not None, np.ndarray[double, ndim=1, mode='c'] result not None):
    cdef int vs0, ms0, ms1
    vs0=vector.shape[0]
    ms0=matrix.shape[0]
    ms1=matrix.shape[1]
    dot_transpose_simple(& vector[0], vs0, & matrix[0, 0], ms0, ms1, & result[0])


@cython.boundscheck(False)
def dot_transpose_mul_c(np.ndarray[double, ndim=1, mode='c'] mult not None, np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=2, mode='c'] matrix not None, np.ndarray[double, ndim=1, mode='c'] result not None):
    cdef int vs0, ms0, ms1
    cdef double ret
    vs0=vector.shape[0]
    ms0=matrix.shape[0]
    ms1=matrix.shape[1]
    dot_transpose(& mult[0], & vector[0], vs0, & matrix[0, 0], ms0, ms1, & result[0])


@cython.boundscheck(False)
def generalized_outer_cython(double alpha,
                             np.ndarray[double, ndim=1, mode='c'] vector1 not None,
                             np.ndarray[double, ndim=1, mode='c'] vector2 not None,
                             double beta,
                             np.ndarray[double, ndim=2, mode='c'] matrix not None,
                             np.ndarray[double, ndim=2, mode='c'] result not None
                             ):
    result[:]=alpha*np.outer(vector1, vector2)+beta*matrix


@cython.boundscheck(False)
def generalized_outer_c(double alpha,
                        np.ndarray[double, ndim=1, mode='c'] vector1 not None,
                        np.ndarray[double, ndim=1, mode='c'] vector2 not None,
                        double beta,
                        np.ndarray[double, ndim=2, mode='c'] matrix not None,
                        np.ndarray[double, ndim=2, mode='c'] result not None
                        ):
    cdef int vs0, vs1
    vs0=vector1.shape[0]
    vs1=vector2.shape[0]
    generalized_outer(alpha, & vector1[0], vs0, & vector2[0], vs1, beta, & matrix[0, 0], & result[0, 0])


@cython.boundscheck(False)
def dot_sigmoid_cython(np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=2, mode='c'] matrix not None, double beta, np.ndarray[double, ndim=1, mode='c'] result not None, int append_bias):
    if append_bias!=0:
        vector = np.append(vector, 1.0)
    result[:]=1.0/(1.0+np.exp(-beta*np.dot(vector, matrix)))


@cython.boundscheck(False)
def dot_sigmoid_c(np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=2, mode='c'] matrix not None, double beta, np.ndarray[double, ndim=1, mode='c'] result not None, int append_bias):
    cdef int ms0, ms1
    ms0 = matrix.shape[0]
    ms1 = matrix.shape[1]
    dot_sigmoid(& vector[0], & matrix[0, 0], ms0, ms1, beta, & result[0], append_bias)


@cython.boundscheck(False)
def dot_sigmoid_c_poly(np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=2, mode='c'] matrix not None, double beta, np.ndarray[double, ndim=1, mode='c'] result not None, int append_bias):
    cdef int ms0, ms1
    ms0 = matrix.shape[0]
    ms1 = matrix.shape[1]
    dot_sigmoid_poly(& vector[0], & matrix[0, 0], ms0, ms1, beta, & result[0], append_bias)


@cython.boundscheck(False)
def dot_add_c(np.ndarray[double, ndim=1, mode='c'] vector not None, np.ndarray[double, ndim=2, mode='c'] matrix not None, np.ndarray[double, ndim=1, mode='c'] result not None, int append_bias):
    cdef int ms0, ms1
    ms0 = matrix.shape[0]
    ms1 = matrix.shape[1]
    dot_add(& vector[0], & matrix[0, 0], ms0, ms1, & result[0], append_bias)


@cython.boundscheck(False)
def sigmoid_poly_c(np.ndarray[double, ndim=1, mode='c'] vector not None, double beta):
    cdef int vs0
    vs0 = vector.shape[0]
    sigmoid_poly(& vector[0], vs0, beta)


@cython.boundscheck(False)
def sigmoid_c(np.ndarray[double, ndim=1, mode='c'] vector not None, double beta):
    cdef int vs0
    vs0 = vector.shape[0]
    sigmoid(& vector[0], vs0, beta)


@cython.boundscheck(False)
def flip_some_bytes(int count, np.ndarray[np.uint8_t, ndim=2] arena, np.ndarray[np.int_t, ndim=1] mrange):
    for i in xrange(count):
        x = np.random.randint(0, mrange[2])
        y = np.random.randint(0, mrange[3])
        arena[mrange[0]+x, mrange[1]+y]=255-arena[mrange[0]+x, mrange[1]+y]


cdef inline int clip(int x, int low, int high):
    if x<low:
        return low
    if x>high:
        return high
    return x


@cython.boundscheck(False)
def ising_model(int count, np.ndarray[np.int8_t, ndim=2] arena, np.ndarray[np.int_t, ndim=1] mrange, float beta):
    cdef int xmax = arena.shape[0]-2
    cdef int ymax = arena.shape[1]-2
    cdef int x = 0
    cdef int y = 0
    cdef int s = 0
    cdef int de = 0
    for i in xrange(count):
        x = clip(mrange[0]+np.random.randint(0, mrange[2]), 1, xmax)
        y = clip(mrange[1]+np.random.randint(0, mrange[3]), 1, ymax)
        s = arena[x-1, y]+arena[x+1, y]+arena[x, y-1]+arena[x, y+1]
        de = s*arena[x, y]-s*(-arena[x, y])
        if de < 0:
            arena[x, y] =- arena[x, y]
        elif np.random.rand()<np.exp(-beta*np.abs(de)):
            arena[x, y] =- arena[x, y]


@cython.boundscheck(False)
def mlpfwd(inputs, hidden, weights, n_layers, beta):
    """
    Run the network forward. This method should not be used externally.
    """
    dot_sigmoid_c(inputs, weights[0], beta, hidden[0], 1)
    hidden[0][-1]=1.0
    for i in xrange(1, n_layers-2):
            dot_sigmoid_c(hidden[i-1], weights[i], beta, hidden[i], 0)
            hidden[i][-1]=1.0
    outputs = np.zeros((weights[-1].shape[1]+1,))
    dot_sigmoid_c(hidden[-1], weights[-1], beta, outputs, 0)
    return outputs[:-1]


@cython.boundscheck(False)
def train(inputs, targets, eta, outputs, hidden, weights, n_layers, beta, momentum, deltas, delta_w, __prev_update):
    """
    Train one sample of data

    """
    outputs[:] = mlpfwd(inputs, hidden, weights, n_layers, beta)
    deltas[n_layers-2][:] = ((targets-outputs)*outputs*(1.0-outputs))
    dot_transpose_mul_c(hidden[n_layers-3]*(1.0-hidden[n_layers-3]), deltas[n_layers-2], weights[n_layers-2], deltas[n_layers-3])
    for i in range(2, n_layers-1):
        j = n_layers-2-i
        dot_transpose_mul_c(hidden[j]*(1.0-hidden[j]), deltas[j+1][:-1], weights[j+1], deltas[j])

    generalized_outer_c(eta, inputs, deltas[0][:-1], momentum, __prev_update[0], delta_w[0])
    for i in range(1, n_layers-2):
        generalized_outer_c(eta, hidden[i-1], deltas[i][:-1], momentum, __prev_update[i], delta_w[i])

    generalized_outer_c(eta, hidden[n_layers-3], deltas[n_layers-2], momentum, __prev_update[n_layers-2], delta_w[n_layers-2])

    for i in range(n_layers-1):
        weights[i] += delta_w[i]
        __prev_update[i][:] = delta_w[i][:]
    return outputs


@cython.boundscheck(False)
def train2(inputs, error, eta, outputs, hidden, weights, n_layers, beta, momentum, deltas, delta_w, __prev_update):
    """
    Train one sample of data

    """
    deltas[n_layers-2][:] = ((error)*outputs*(1.0-outputs))
    dot_transpose_mul_c(hidden[n_layers-3]*(1.0-hidden[n_layers-3]), deltas[n_layers-2], weights[n_layers-2], deltas[n_layers-3])
    for i in range(2, n_layers-1):
        j = n_layers-2-i
        dot_transpose_mul_c(hidden[j]*(1.0-hidden[j]), deltas[j+1][:-1], weights[j+1], deltas[j])

    generalized_outer_c(eta, inputs, deltas[0][:-1], momentum, __prev_update[0], delta_w[0])
    for i in range(1, n_layers-2):
        generalized_outer_c(eta, hidden[i-1], deltas[i][:-1], momentum, __prev_update[i], delta_w[i])
    generalized_outer_c(eta, hidden[n_layers-3], deltas[n_layers-2], momentum, __prev_update[n_layers-2], delta_w[n_layers-2])
    for i in range(n_layers-1):
        weights[i] += delta_w[i]
        __prev_update[i][:] = delta_w[i][:]
    return outputs
