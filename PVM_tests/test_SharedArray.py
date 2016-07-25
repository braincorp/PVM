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


import pytest
import PVM_framework.SharedArray as SharedArray
import numpy as np
import cPickle
import gzip
import multiprocessing


def save(pobject, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    pfile = gzip.GzipFile(filename, 'wb')
    cPickle.dump(pobject, pfile, protocol)
    pfile.close()


def load(filename):
    """Loads a compressed object from disk
    """
    pfile = gzip.GzipFile(filename, 'rb')
    prop_dict = cPickle.load(pfile)
    pfile.close()
    return prop_dict


def test_pickling_array(tmpdir):
    """
    Pickles and unpickles a shared array
    """
    A = SharedArray.SharedNumpyArray((10, 10), np.float)
    A[1, 1] = 15
    save(A, str(tmpdir)+"/test.p.gz")
    B = load(str(tmpdir)+"/test.p.gz")
    assert(B[1, 1] == 15)


def test_pickling_darray(tmpdir):
    """
    Pickles and unpickles a double buffered shared array
    """
    parity = SharedArray.SharedNumpyArray((1,), np.uint32)
    A = SharedArray.DoubleBufferedSharedNumpyArray((10, 10), np.float, parity)
    A[1, 1] = 15
    parity[0] = 1
    A[1, 1] = 12
    save({"array": A, "parity": parity}, str(tmpdir)+"/test.p.gz")
    ddict = load(str(tmpdir)+"/test.p.gz")
    B = ddict["array"]
    parity = ddict["parity"]
    assert(B[1, 1] == 12)
    parity[0] = 0
    assert(B[1, 1] == 15)
    

def worker_test_sharednes_of_an_array(Array, A):
    """
    Worker code spawned in a separate process to test
    if the shared array is in fact shared.
    """
    Array[5, 5] = 10
    return


def test_sharedness_of_an_array():
    """
    Test if the array is actually shared between two processes.
    """
    for dtype in (np.float, np.float16, np.float32, np.float64, np.float128, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.complex, np.complex64):
        Array = SharedArray.SharedNumpyArray((10, 10), dtype)
        p = multiprocessing.Process(target=worker_test_sharednes_of_an_array, args=(Array, None))
        p.start()
        p.join()
        assert(Array[5, 5] == 10)

 
def worker_test_sharedness_of_a_darray(Array, Parity):
    """
    Worker code spawned in a separate process to test
    if the double buffered shared array is in fact shared.
    """    
    Array[5, 5] = 10
    Parity[0] = 1
    Array[5, 5] = 15
    return


def test_sharedness_of_a_darray():
    """
    Test if the double buffered array is actually shared between two processes.
    """
    parity = SharedArray.SharedNumpyArray((1,), np.uint32)
    Array = SharedArray.DoubleBufferedSharedNumpyArray((10, 10), np.float, parity)
    p = multiprocessing.Process(target=worker_test_sharedness_of_a_darray, args=(Array, parity))
    p.start()
    p.join()
    assert(Array[5, 5] == 15)
    parity[0] = 0
    assert(Array[5, 5] == 10)


def worker_test_IPC_attachability():
    Array=SharedArray.SharedNumpyArray((10, 10), np.float, tag="abcdefgh312", create=False)
    Array[5, 5] = 10
    return


def test_IPC_attachability():
    """
    Test if the array is actually shared between two processes.
    """
    Array=SharedArray.SharedNumpyArray((10, 10), np.float, tag="abcdefgh312", create=True)
    p = multiprocessing.Process(target=worker_test_IPC_attachability, args=())
    p.start()
    p.join()
    assert(Array[5, 5] == 10)


if __name__ == "__main__":
    manual_test = True
    if manual_test:
        test_pickling_array()
        test_pickling_darray()
        test_sharedness_of_an_array()
        test_sharedness_of_a_darray()
        test_IPC_attachability()
    else:
        pytest.main('-s %s -n0' % __file__)
