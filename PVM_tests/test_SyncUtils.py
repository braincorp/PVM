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
import multiprocessing
test_spinlocks = False
try:
    import PVM_framework.SyncUtils as SyncUtils
    test_spinlocks = True
except:
    import PVM_framework.SyncUtils_python as SyncUtils
import py


def ver_to_float(ver):
    nums = ver.split('.')
    result = 0.0
    base = 1.0
    for num in nums:
        result += float(num)*base
        base *= 0.001
    return result


def worker_test_barrier(barrier, array, proc_id):
    """
    Worker code, spawn in many copies as separate processes
    """
    barrier.worker_barrier()
    condition = True
    while condition:
        array[proc_id] += 1
        condition = barrier.worker_barrier()


def parrent_test_barrier(UseSpinLock=False):
    """
    Parent (controller) process, instantiates the shared memory and 
    synchronization objects and spawns worker processes. Verifies that
    the data consistency is assured as the code executes.
    """
    test_barrier = SyncUtils.Barrier(4, timeout=0.001, UseSpinLock=UseSpinLock)
    shared_data = SharedArray.SharedNumpyArray((4,), np.float)
    procs = []
    for i in xrange(4):
        proc = multiprocessing.Process(target=worker_test_barrier, args=(test_barrier, shared_data, i))
        procs.append(proc)
        proc.start()
    test_barrier.parent_barrier()
    test_barrier.resume_workers()
    for i in xrange(4):
        test_barrier.parent_barrier()
        assert (sum(shared_data) == (i + 1) * 4)
        test_barrier.resume_workers()
    test_barrier.parent_barrier(quit_workers=True)
    for p in procs:
        p.join()


def test_barrier():
    """
    Test barrier with regular locks provided by the operating system.
    """
    parrent_test_barrier(UseSpinLock=False)


def test_barrier_spinlocks():
    """
    Test barrier with spin locks implemented using atomic operation on a shared
    array object (busy wait). Use spinlocks for highly parallel applications (>10 worker processes)
    where synchronization lag needs to be minimal even the expense of some busy wait.
    """
    if (not test_spinlocks):
        py.test.skip("gcc > 4.6 is nescessary for this test")
    parrent_test_barrier(UseSpinLock=True)


if __name__ == "__main__":
    manual_test = True
    if manual_test:
        test_barrier()
        test_barrier_spinlocks()
    else:
        pytest.main('-s %s -n0' % __file__)
