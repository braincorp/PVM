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
from multiprocessing.sharedctypes import RawArray
import sys
import mmap
import ctypes
import posix_ipc
from _multiprocessing import address_of_buffer
from string import ascii_letters, digits
import pickle

valid_chars = frozenset("-_. %s%s" % (ascii_letters, digits))


class ShmemBufferWrapper(object):
    """
    IPC shared memory buffer wrapper.
    """
    def __init__(self, tag, size, create=True):
        self._mem = None
        self._map = None
        self._owner = create
        self.size = size

        assert 0 <= size < sys.maxint
        flag = (0, posix_ipc.O_CREX)[create]
        if create:
            self._mem = posix_ipc.SharedMemory(tag, flags=flag, size=size)
        else:
            self._mem = posix_ipc.SharedMemory(tag, flags=flag, size=0)
        self._map = mmap.mmap(self._mem.fd, self._mem.size)
        self._mem.close_fd()

    def get_address(self):
        addr, size = address_of_buffer(self._map)
        assert size >= self.size
        return addr

    def __del__(self):
        if self._map is not None:
            self._map.close()
        if self._mem is not None and self._owner:
            self._mem.unlink()


def ShmemRawArray(type_, size_or_initializer, tag, create=True):
    """
    Raw shared memory array based on IPC tag
    :param type_:
    :param size_or_initializer:
    :param tag:
    :param create:
    :return:
    """
    if tag[0] != "/":
        tag = "/%s" % (tag,)

    if isinstance(size_or_initializer, int):
        type_ = type_ * size_or_initializer
    else:
        type_ = type_ * len(size_or_initializer)

    buffer = ShmemBufferWrapper(tag, ctypes.sizeof(type_), create=create)
    obj = type_.from_address(buffer.get_address())
    obj._buffer = buffer

    if not isinstance(size_or_initializer, int):
        obj.__init__(*size_or_initializer)

    return obj


def np_type_id_to_ctypes(dtype):
    type_id = None
    if hasattr(np, 'float') and dtype == np.float:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'float16') and dtype == np.float16:
        type_id = ctypes.c_int16
        return type_id
    if hasattr(np, 'float32') and dtype == np.float32:
        type_id = ctypes.c_int32
        return type_id
    if hasattr(np, 'float64') and dtype == np.float64:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'float128') and dtype == np.float128:
        type_id = (ctypes.c_int64 * 2)
        return type_id
    if hasattr(np, 'int') and dtype == np.int:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'uint8') and dtype == np.uint8:
        type_id = ctypes.c_byte
        return type_id
    if hasattr(np, 'int8') and dtype == np.int8:
        type_id = ctypes.c_byte
        return type_id
    if hasattr(np, 'uint16') and dtype == np.uint16:
        type_id = ctypes.c_int16
        return type_id
    if hasattr(np, 'int16') and dtype == np.int16:
        type_id = ctypes.c_int16
        return type_id
    if hasattr(np, 'uint32') and dtype == np.uint32:
        type_id = ctypes.c_int32
        return type_id
    if hasattr(np, 'int32') and dtype == np.int32:
        type_id = ctypes.c_int32
        return type_id
    if hasattr(np, 'uint64') and dtype == np.uint64:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'int64') and dtype == np.int64:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'complex') and dtype == np.complex:
        type_id = (ctypes.c_int64 * 2)
        return type_id
    if hasattr(np, 'complex64') and dtype == np.complex64:
        type_id = (ctypes.c_int64)
        return type_id
    if hasattr(np, 'intc') and dtype == np.intc:
        type_id = (ctypes.c_int)
        return type_id
    if hasattr(np, 'intp') and dtype == np.intp:
        type_id = (ctypes.c_ssize_t)
        return type_id
    if hasattr(np, 'bool') and dtype == np.bool:
        type_id = (ctypes.c_byte)
        return type_id
    raise Exception('No matching data type!')


class SharedNumpyArray:  # DO NOT EXTEND FROM OBJECT!
    """
    Acts like a numpy array but is shared!
    One notable difference - added copyto method which copies from a given object
    and places the content in the shared memory. The reason for that is
    that numpy.copyto will not work on this object directly unfortunately.
    Uses raw array so the if you need a lock you have to instantiate it yourself.

    If given a tag in the constructor, will actually generate a POSIX IPC memory
    that can be attached from a different process. Note that if this is the case,
    the array is no longer serializable.
    """
    def __init__(self, shape, dtype, tag=None, create=True):
        type_id = np_type_id_to_ctypes(dtype)
        self.tag = tag
        if tag is not None:
            self.__shared = ShmemRawArray(type_id, np.product(shape), tag, create=create)
        else:
            self.__shared = RawArray(type_id, np.product(shape))
        self.__np_array = np.frombuffer(self.__shared, dtype=dtype).reshape(shape)

    def __getattr__(self, name):
        """
        This is only called if python cannot find the attribute. This is the reason
        why this class cannot be derived from object which comes with a bunch of attributes
        and screws this thing up.
        """
        return self.__np_array.__getattribute__(name)

    def __getstate__(self):
        """
        Method overloaded for support of pickling.
        """
        if self.tag is not None:
            raise pickle.PicklingError()
        state = self.__dict__.copy()
        del state['_SharedNumpyArray__shared']
        return state

    def __setstate__(self, state):
        """
        Method overloaded for support of pickling.
        """
        shape = state['_SharedNumpyArray__np_array'].shape
        dtype = state['_SharedNumpyArray__np_array'].dtype
        type_id = np_type_id_to_ctypes(dtype)
        self.__shared = RawArray(type_id, np.product(shape))
        self.__np_array = np.frombuffer(self.__shared, dtype=dtype).reshape(shape)
        np.copyto(self.__np_array, state['_SharedNumpyArray__np_array'])
        self.tag = None

    def copyto(self, nparray):
        """
        Implemented because numpy.copyto will not work directly on this object
        """
        np.copyto(self.__np_array, nparray)


class DynamicView:
    """
    Picklable view object. A view similar to a pointer offers an array like object
    whose content coresponds to a subarray of some existing array. Normal views on shared numpy
    arrays are not picklable (after unpickling they will recrate a copy of the original array). Dynamic
    view is fully picklable.
    """
    def __init__(self, array):
        self._array = array
        self._view = self._array
        self._view_item = None
        self._version = 0.1

    def __getitem__(self, item):
        if type(item) == slice or (type(item) == tuple) or type(item) == int:
            self._view_item = item
            self._view = self._array.__getitem__(item)
        return self

    def __getattr__(self, name):
        """
        This is only called if python cannot find the attribute. This is the reason
        why this class cannot be derived from object which comes with a bunch of attributes
        and screws this thing up.
        """
        return self._view.__getattribute__(name)

    def __getstate__(self):
        """
        Method overloaded for support of pickling.
        """
        state = self.__dict__.copy()
        del state['_view']
        return state

    def __setstate__(self, state):
        """
        Method overloaded for support of pickling.
        """
        if "_version" not in state.keys():  # Compatibility mode
            self._array = state['_array']
            self._version = 0.1  # promote to the latest version
            x = state['_x']
            y = state['_y']
            z = state['_z']
            dx = state['_dx']
            dy = state['_dy']
            dz = state['_dz']
            if x is not None and y is not None and z is not None:
                self[x:x+dx, y:y+dy, z:z+dz]
            elif x is not None and y is not None:
                self[x:x+dx, y:y+dy]
            elif x is not None:
                self[x:x+dx]
            else:
                self._view = self._array
                self._view_item = None
        else:
            self.__dict__ = state
            if self._view_item is not None:
                self.__getitem__(self._view_item)
            else:
                self._view = self._array


def SharedNumpyArray_like(array):
    """
    Creates a shared array object identical as the given numpy array (but without the content)
    :param array: numpy array object
    :return: shared array object
    """
    return SharedNumpyArray(shape=array.shape, dtype=array.dtype)


class DoubleBufferedSharedNumpyArray:  # DO NOT EXTEND FROM OBJECT!
    """
    Acts like a numpy array but is shared!
    One notable difference - added copyto method which copies from a given object
    and places the content in the shared memory. The reason for that is
    that numpy.copyto will not work on this object directly unfortunately.
    Uses raw array so the if you need a lock you have to instantiate it yourself.
    """
    def __init__(self, shape, dtype, parity_obj):
        type_id = np_type_id_to_ctypes(dtype)

        self.__shared1 = RawArray(type_id, np.product(shape))
        self.__np_array1 = np.frombuffer(self.__shared1, dtype=dtype).reshape(shape)
        self.__shared2 = RawArray(type_id, np.product(shape))
        self.__np_array2 = np.frombuffer(self.__shared2, dtype=dtype).reshape(shape)
        self.__parity = parity_obj

    def __getattr__(self, name):
        if self.__parity[0]==0:
            return self.__np_array1.__getattribute__(name)
        else:
            return self.__np_array2.__getattribute__(name)

    def __getstate__(self):
        """
        Method overloaded for support of pickling.
        """
        state = self.__dict__.copy()
        del state['_DoubleBufferedSharedNumpyArray__shared1']
        del state['_DoubleBufferedSharedNumpyArray__shared2']
        return state

    def __setstate__(self, state):
        """
        Method overloaded for support of pickling.
        """
        shape = state['_DoubleBufferedSharedNumpyArray__np_array1'].shape
        dtype = state['_DoubleBufferedSharedNumpyArray__np_array1'].dtype
        type_id = np_type_id_to_ctypes(dtype)
        self.__shared1 = RawArray(type_id, np.product(shape))
        self.__np_array1 = np.frombuffer(self.__shared1, dtype=dtype).reshape(shape)
        np.copyto(self.__np_array1, state['_DoubleBufferedSharedNumpyArray__np_array1'])
        self.__shared2 = RawArray(type_id, np.product(shape))
        self.__np_array2 = np.frombuffer(self.__shared2, dtype=dtype).reshape(shape)
        np.copyto(self.__np_array2, state['_DoubleBufferedSharedNumpyArray__np_array2'])
        self.__parity = state['_DoubleBufferedSharedNumpyArray__parity']

    def copyto(self, nparray):
        """
        Implemented because numpy.copyto will not work directly on this object
        """
        if self.__parity[0]==0:
            np.copyto(self.__np_array1, nparray)
        else:
            np.copyto(self.__np_array2, nparray)

    def copytobuffer(self, nparray):
        """
        Implemented because numpy.copyto will not work directly on this object
        """
        if self.__parity[0]==0:
            np.copyto(self.__np_array2, nparray)
        else:
            np.copyto(self.__np_array1, nparray)
