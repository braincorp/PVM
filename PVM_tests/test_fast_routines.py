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
import PVM_framework.fast_routines as fast_routines
import numpy as np
import time


def test_derivative_dot():
    print "------ derivative_dot ------"
    vector = np.random.randn(5000)
    vector2 = np.ones((5000,))
    result_cython = np.zeros((5000,))
    result_c = np.zeros((5000,))
    t_start = time.time()
    fast_routines.derivative_dot_cython(vector, vector2, result_cython)
    cython_time=time.time()-t_start
    print "Cython implementation %f" % (cython_time)
    t_start = time.time()
    fast_routines.derivative_dot_c(vector, vector2, result_c)
    c_time=time.time()-t_start
    print "C implementation %f " % (c_time)
    print "Acceleration %f" % (cython_time/c_time)
    assert(np.allclose(result_cython, result_c))


def test_dot_transpose():
    print "------ dot_transpose ------"
    vector = np.random.randn(30000)
    matrix = np.random.randn(5000, 30000)
    multiplier=np.ones((5000,))
    result_cython=np.zeros((5000,))
    result_c=np.zeros((5000,))
    t_start = time.time()
    fast_routines.dot_transpose_mul_cython(multiplier, vector, matrix, result_cython)
    cython_time=time.time()-t_start
    print "Cython implementation %f" % (cython_time)
    t_start = time.time()
    fast_routines.dot_transpose_mul_c(multiplier, vector, matrix, result_c)
    c_time=time.time()-t_start
    print "C implementation %f " % (c_time)
    print "Acceleration %f" % (cython_time/c_time)
    assert(np.allclose(result_cython, result_c))


def test_dot_transpose_simple():
    print "------ dot_transpose_simple ------"
    vector = np.random.randn(30000)
    matrix = np.random.randn(5000, 30000)
    result_cython=np.zeros((5000,))
    result_c=np.zeros((5000,))
    t_start = time.time()
    fast_routines.dot_transpose_cython(vector, matrix, result_cython)
    cython_time=time.time()-t_start
    print "Cython implementation %f" % (cython_time)
    t_start = time.time()
    fast_routines.dot_transpose_c(vector, matrix, result_c)
    c_time=time.time()-t_start
    print "C implementation %f " % (c_time)
    print "Acceleration %f" % (cython_time/c_time)
    assert(np.allclose(result_cython, result_c))


def test_generalized_outer():
    print "------ generalized outer ------"
    d1=4000
    d2=5000
    vector1 = np.random.randn(d1)
    vector2 = np.random.randn(d2)
    matrix = np.random.randn(d1, d2)
    result_cython = np.zeros((d1, d2))
    result_c = np.zeros((d1, d2))
    t_start = time.time()
    fast_routines.generalized_outer_cython(1.0, vector1, vector2, 0.5, matrix, result_cython)
    cython_time = time.time()-t_start
    print "Cython implementation %f" % cython_time
    t_start = time.time()
    fast_routines.generalized_outer_c(1.0, vector1, vector2, 0.5, matrix, result_c)
    c_time=time.time()-t_start
    print "C implementation %f" % (c_time)
    print "Acceleration %f" % (cython_time/c_time)
    assert(np.allclose(result_cython, result_c))


def dot_sigmoid(append_bias):
    print "------ dot_sigmoid ------"
    d1=500
    d2=400
    vector1 = np.random.randn(d1)
    matrix = np.random.randn(d1+append_bias, d2)
    result_cython = np.zeros((d2))
    result_c = np.zeros((d2))
    t_start = time.time()
    fast_routines.dot_sigmoid_cython(vector1, matrix, 1.0, result_cython, append_bias)
    cython_time = time.time()-t_start
    print "Cython implementation %f" % cython_time
    t_start = time.time()
    fast_routines.dot_sigmoid_c(vector1, matrix, 1.0, result_c, append_bias)
    c_time=time.time()-t_start
    print "C implementation %f" % (c_time)
    print "Acceleration %f" % (cython_time/c_time)
    assert(np.allclose(result_cython, result_c))


def test_dot_sigmoid():
    dot_sigmoid(0)
    dot_sigmoid(1)


def test_dot_add():
    d1=500
    d2=400
    vector1 = np.random.randn(d1)
    matrix = np.random.randn(d1, d2)
    result_c = np.zeros((d2))
    t_start = time.time()
    result_cython=np.dot(matrix.T, vector1)
    cython_time = time.time()-t_start
    print "Cython implementation %f" % cython_time
    t_start = time.time()
    fast_routines.dot_add_c(vector1, matrix, result_c, 0)
    c_time=time.time()-t_start
    print "C implementation %f" % (c_time)
    print "Acceleration %f" % (cython_time/c_time)
    assert(np.allclose(result_cython, result_c))


if __name__=="__main__":
    test_dot_transpose()
    test_generalized_outer()
    test_dot_sigmoid()
