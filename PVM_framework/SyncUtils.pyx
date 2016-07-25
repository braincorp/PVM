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
from multiprocessing.sharedctypes import RawArray
import ctypes
from cpython.buffer cimport PyBUF_SIMPLE
from cpython.buffer cimport Py_buffer
from cpython.buffer cimport PyObject_GetBuffer
from cpython.buffer cimport PyBuffer_Release
import time
import multiprocessing as mp


cdef extern from "Sync.h":
    cdef long atomic_get_and_set(long n, long*)
    cdef void acquire(long*)
    cdef void release(long*)
    cdef void flush_denormals()


cdef class SpinLock:
    cdef long* addr
    cdef int key
    cdef object array

    def __init__(self):
        self.array = RawArray(ctypes.c_int64, 1)
        cdef Py_buffer view
        PyObject_GetBuffer(self.array, &view, PyBUF_SIMPLE)
        self.addr = <long *> view.buf

    def acquire(self):
        acquire(self.addr)

    def release(self):
        release(self.addr)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, dtype, value, traceback):
        return self.release()


cdef class SpinLockRaw:
    cdef long* addr
    cdef int key
    cdef object array

    def __init__(self):
        self.array = RawArray(ctypes.c_int64, 1)
        cdef Py_buffer view
        PyObject_GetBuffer(self.array, &view, PyBUF_SIMPLE)
        self.addr = <long *> view.buf

    def acquire(self):
        acquire(self.addr)

    def release(self):
        release(self.addr)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, dtype, value, traceback):
        return self.release()

cdef class Barrier:
    """
    Object meant to provide barrier functionality. Instantiated in the parent process,
    can be passed onto child processes and used for synchronization. Relies on multiprocessing
    Value and Lock.
    """
    cdef object worker_procs
    cdef object workers_waiting
    cdef object workers_ready
    cdef object parent_resumed
    cdef object __quit_workers
    cdef object worker_procs_lock
    cdef object workers_waiting_lock
    cdef object workers_ready_lock
    cdef object __timeout

    def __init__(self, num_proc, timeout = 0, UseSpinLock=True):
        self.worker_procs = mp.sharedctypes.RawValue('i')
        self.worker_procs.value = num_proc
        self.workers_waiting = mp.sharedctypes.RawValue('i')
        self.workers_ready = mp.sharedctypes.RawValue('i')
        self.parent_resumed = mp.sharedctypes.RawValue('i')
        self.__quit_workers = mp.sharedctypes.RawValue('i')
        if UseSpinLock:
            self.worker_procs_lock = SpinLock()
            self.workers_waiting_lock = SpinLock()
            self.workers_ready_lock = SpinLock()
        else:
            self.worker_procs_lock = mp.Lock()
            self.workers_waiting_lock = mp.Lock()
            self.workers_ready_lock = mp.Lock()
        self.__timeout = timeout

    def __inc(self, val, lock):
        lock.acquire()
        val.value+=1
        lock.release()

    def __check_and_reset(self, val, lock, check_val):
        if val.value == check_val:
            lock.acquire()
            val.value = 0
            lock.release()
            return True
        return False

    def worker_barrier(self):
        """
        The call will block the execution on the worker side until all the workers
        reach barrier_worker and the parent process reaches resume_workers
        """
        self.__inc(self.workers_waiting, self.workers_waiting_lock)
        while self.workers_waiting.value != 0:
            if self.__timeout>0:
                time.sleep(self.__timeout)
        while self.parent_resumed.value == 0:
            if self.__timeout>0:
                time.sleep(self.__timeout)
        if self.__quit_workers.value == 1:
            self.worker_procs_lock.acquire()
            self.worker_procs.value-=1
            self.worker_procs_lock.release()
            return False
        self.__inc(self.workers_ready, self.workers_ready_lock)
        while self.workers_ready.value != 0:
            if self.__timeout>0:
                time.sleep(self.__timeout)
        if self.__quit_workers.value == 1:
            self.worker_procs_lock.acquire()
            self.worker_procs.value-=1
            self.worker_procs_lock.release()
            return False
        return True

    def parent_barrier(self, quit_workers=False):
        """
        Wait until all the workers reach barrier_worker. Resume once they do.
        """
        self.parent_resumed.value = 0
        while not self.__check_and_reset(self.workers_waiting, self.workers_waiting_lock, self.worker_procs.value):
            if self.__timeout>0:
                time.sleep(self.__timeout)
        if quit_workers:
            self.__quit_workers.value = 1
        self.parent_resumed.value = 1

    def resume_workers(self, quit_workers=False):
        """
        Notify the worker processes to resume from barrier_worker
        """
        if quit_workers:
            self.__quit_workers.value = 1
        while not self.__check_and_reset(self.workers_ready, self.workers_ready_lock, self.worker_procs.value):
            if self.__timeout>0:
                time.sleep(self.__timeout)

    def quit_workers(self):
        self.parent_barrier(quit_workers=True)

    def workers_running(self):
        """
        Return the number of active workers
        """
        return self.worker_procs.value
