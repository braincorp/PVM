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

import cPickle
import gzip
import sys
import time
import multiprocessing as mp
import socket
import threading
import PVM_framework.PVM_debug_console as PVM_debug_console
import os
import debug_logger
import logging
import glob
import importlib
import random
import datetime
try:
    import PVM_framework.SyncUtils as SyncUtils
except:
    import PVM_framework.SyncUtils_python as SyncUtils
import pickle
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

# Legacy class/module renaming
renametable = {'future_encoder_framework.SharedArray': 'PVM_framework.SharedArray',
               'PVM_tools.labeled_movie': 'PVM_tools.labeled_movie',
               'PVM_tools.bounding_region': 'PVM_tools.bounding_region',
               }


def mapname(name):
    if name in renametable:
        return renametable[name]
    return name


def mapped_load_global(self):
    module = mapname(self.readline()[:-1])
    name = mapname(self.readline()[:-1])
    klass = self.find_class(module, name)
    self.append(klass)


def load_legacy_pickle(str):
    file = StringIO(str)
    unpickler = pickle.Unpickler(file)
    unpickler.dispatch[pickle.GLOBAL] = mapped_load_global
    return unpickler.load()


def load_legacy_pickle_file(file):
    unpickler = pickle.Unpickler(file)
    unpickler.dispatch[pickle.GLOBAL] = mapped_load_global
    return unpickler.load()


def save_model(pobject, filename, protocol=-1):
    """
    Save an object to a compressed disk file.

    :param pobject: serializable python object
    :type pobject: object
    :param filename: string representing a path to a file
    :type filename: str
    :param protocol: optional protocol parameter (defaults to -1 which is the latest available)
    :type protocol: int
    """
    logging.info("Saving a model to a file " + str(filename))
    pfile = gzip.GzipFile(filename, 'wb')
    cPickle.dump(pobject, pfile, protocol)
    pfile.close()
    logging.info("Saved a model to a file " + str(filename))
    logging.info("Warning: Saved to a python pickle. Note, if you ever use this code for production consider using other "
                 "serializing methods. Pickle is python specific and may pose a security threat")


def load_model(filename):
    """
    Loads a compressed object from disk

    :param filename: string representing a path to a file
    :type filename: str
    :return: python object loaded from file
    :rtype: object
    """
    logging.info("Loading a model from a file " + str(filename))
    pfile = gzip.GzipFile(filename, 'rb')
    prop_dict = load_legacy_pickle(pfile.read())
    pfile.close()
    logging.info("Loaded a model from a file " + str(filename))
    logging.info("Warning: Loaded from a python pickle. Note, if you ever use this code for production consider using other "
                 "serializing methods. Pickle is python specific and may pose a security threat")
    return prop_dict


def _worker_code(prop_dict, proc_id, barrier):
    """
    The code executed by a single worker process, instantiates and
    executes a subset of all the objects described in the dictionary,
    synchronizes with other workers and the supervisor thread.

    :param prop_dict: Dictionary with simulation data
    :param proc_id: process id provided by the parrent (not the unix pid)
    :param barrier: barrier object used for synchronization
    :return:
    """
    try:
        import PVM_framework.SyncUtils as SyncUtils
    except:
        import PVM_framework.SyncUtils_python as SyncUtils
    # Set the CPU flags
    try:
        import PVM_framework.LowLevelCPUControlx86 as LowLevelCPUControlx86
        LowLevelCPUControlx86.set_flush_denormals()
        # Flush denormals will collapse all floating point operation results
        # to zero if they are to small to fit in the normal float representation.
        # Although such action may slighly affect the precission of calculations
        # on some processors (e.g. x86) it substantially speeds up the execution
    except:
        # Unable to set the flags
        print "Setting flush denormals CPU flag not available"
    import numpy as np
    np.random.seed(proc_id)
    # Load the execution module
    while True:  # bacause of an aparent race condition, the first load may not be succesfull,
        try:
            unit = importlib.import_module(prop_dict['execution_unit_module'])
        except:
            continue
        break
    steps = int(unit.ExecutionUnit.execution_steps())
    if barrier.worker_barrier() is False:
        sys.exit(0)
    # Instantiate all the objects
    stages = prop_dict['stages']
    objects = {}  # Local objects of the worker
    calls = {}
    max_procs = prop_dict['num_proc']
    for stagenum in xrange(stages):
        stage = 'stage%d' % stagenum
        total_stage_elements = prop_dict['stage%d_size' % stagenum]
        my_stage_elements = range(proc_id, total_stage_elements, max_procs)
        objects[stage] = {}
        calls[stage] = {}
        objects[stage]['my_elements'] = my_stage_elements
        objects[stage]['units'] = []
        for i in my_stage_elements:
            Cunit = unit.ExecutionUnit(parameters=prop_dict[stage][i])
            objects[stage]['units'].append(Cunit)
        calls[stage] = []
        for j in range(steps):
            calls[stage].append([])
            for (i, _) in enumerate(objects[stage]['my_elements']):
                method = getattr(objects[stage]['units'][i], "execute%d" % j)
                calls[stage][j].append(method)

    # The main loop
    quit_now = False
    while True:
        for stagenum in range(stages):
            stage = 'stage%d' % stagenum
            for j in range(steps):
                while prop_dict['paused'][0] == 1:
                    time.sleep(1)
                for method in calls[stage][j]:
                    method()
                if barrier.worker_barrier() is False:
                    quit_now = True
                    break
            if quit_now:
                break
        if quit_now:
            for stagenum in range(stages):
                stage = 'stage%d' % stagenum
                for (i, _) in enumerate(objects[stage]['my_elements']):
                    objects[stage]['units'][i].cleanup()
            sys.exit(0)


def _supervisor_run(prop_dict, control_barrier):
    """
    This is the supervisor (parent class) code. For efficiency reasons the loop
    that synchronized the worker nodes is not run in the main thread. Main thread
    can perform actions like window display etc. in a loosely synchronized manner with the
    rest of the execution.

    [MAIN THREAD]       [ SUPERVISOR ]        [ WORKER 0]   ...
    runs manager          runs the  -------->  executes
    displays windows      main loop.           the unit
    adjust dict params.   Syncs to             code
    Syncs to the          workers  --------->
    supervisor     <----> and main thread

    :param prop_dict: dictionary with simulation memory and parameters
    :param control_barrier: barrier object used for synchronization with the main thread
    :return:
    """
    tryonce = True
    while True:
        try:
            unit = importlib.import_module(prop_dict['execution_unit_module'])
        except:
            if tryonce:
                logging.error("Could not load module %s, going insane in infinite loop. Kill me." % prop_dict['execution_unit_module'])
                tryonce = False
            continue
        break
    logging.info("Loaded the module %s, no reason to kill me yet." % prop_dict['execution_unit_module'])
    steps = unit.ExecutionUnit.execution_steps()
    barrier = SyncUtils.Barrier(prop_dict['num_proc'])
    procs = []
    for i in xrange(prop_dict['num_proc']):
        p = mp.Process(target=_worker_code, args=(prop_dict, i, barrier))
        p.start()
        procs.append(p)
    logging.info("Created workers, no one started yet")
    sys.stdout.flush()
    time.sleep(1)
    sys.stdout.flush()
    barrier.parent_barrier()
    N = prop_dict['N'][0]
    control_barrier.worker_barrier()
    logging.info("Starting workers")
    while barrier.workers_running() > 0:
        # Main execution loop

        for i in range(prop_dict['stages']):
            for j in range(steps):
                barrier.resume_workers()
                while prop_dict['paused'][0] == 1:
                    time.sleep(1)
                barrier.parent_barrier()
        N += 1
        prop_dict['N'][0] = N
        running = control_barrier.worker_barrier()
        if not running:
            prop_dict['paused'][0] = 0
            barrier.resume_workers()
            barrier.parent_barrier(quit_workers=True)
            logging.info("Quitting")
            sys.stdout.flush()
            break
    logging.info("All have finished")
    sys.stdout.flush()
    for p in procs:
        p.join()


def _run_debug_session(socket, prop_dict):
    """
    Starts a debugging session

    :param socket: Network socket on which the session is transmitted
    :param prop_dict: Simulation dictionary of the currently running simulation
    :return:
    """
    file = socket.makefile(mode="rw")
    print >>file, ""
    print >>file, "Predictive Vision Framework version 1.0"
    print >>file, "(C) 2016 Brain Corporation Technical Services"
    print >>file, ""
    print >>file, "You are connected to a debug shell"
    print >>file, "Type 'help' for available commands, 'quit' to exit debug shell"
    explorer = PVM_debug_console.InteractiveDictionaryExplorer(stdin=file, stdout=file, dict=prop_dict, infilename="engine"+str(os.getpid()))
    explorer.cmdloop()
    logging.info("Ended a remote debug session")
    socket.close()


def _monitor_debug_session(port, stop):
    """
    Separate thread code to monitor the debug session and end it if
    the stop() call is true. This is necessary because the debug server might be
    stuck on listen. When the simulation is finished (via any external event) the thread
    processing the server needs to be waken up from the listen mode. The function below accomplishes
    that by periodically trying to connect to it.

    :param port: Port on which the session is running
    :param stop: A function returning a bool value of whether the session should be stopped
    :return:
    """
    while 1:
        time.sleep(0.5)
        if stop():
            S = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            S.connect(("localhost", port))
            S.close()
            logging.info("Closing the server socket")
            break


def _run_debug_server(prop_dict, port, stop):
    """
    Sets up a debug server at a given port

    :param prop_dict: Simulation dictionary of the currently running simulation
    :param port: Port on which to setup the debug server
    :param stop: A function returning a bool value of whether the session should be stopped
    :return:
    """
    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    while True:
        try:
            serversock.bind(("", port))
            break
        except:
            logging.error("Port %d already taken, trying to bind to port %d" % (port, port +1))
            port += 1
            continue
    serversock.listen(5)
    logging.info("Listening on port " + str(port) + " for debug connections")
    clients = []
    monitor = threading.Thread(target=_monitor_debug_session, args=(port, stop))
    monitor.start()
    while 1:
        clientsock, addr = serversock.accept()
        if stop():
            logging.info("Exiting debug session")
            sys.stdout.flush()
            break
        logging.info("Accepted a debug connection from " + str(addr))
        clients.append(threading.Thread(target=_run_debug_session, args=(clientsock, prop_dict)))
        clients[-1].start()
    monitor.join()


class ModelExecution(object):
    """
    ModelExecution is a class allowing to control the execution of a model from the main thread.

    It takes a dictionary and a manager and allows for two modes of operation:

    1. blocking. The start call will block until the execution is complete (as decided by the manager)
    2. non blocking. Allows the main thread to control the execution step by step.

    :param prop_dict: Dictionary containing model data
    :type prop_dict: dict
    :param manager: Manager object controlling the aspects of execution
    :type AbstractExecutionManager: ExecutionManager
    """
    def __init__(self, prop_dict, manager, port=9000):
        self._prop_dict = prop_dict
        self._manager = manager
        self._port = port

    def start(self, blocking=True):
        """
        Begins the execution of the model. If blocking is True, the call will block the main thread
        until the model execution is complete. Otherwise the call will return and allow for stepped
        execution.

        .. note::

            If running in non blocking mode, the call will return, but all the necessary execution threads will
            be running. Since the system uses busy synchronization (Spin Locks) the CPU utilization after the call
            will be maximal. In order to maximize machine utilization, step() should be called as fast as possible.


        :Example:
        ::

            manager = Manager(dict, 1000)
            executor = CoreUtils.ModelExecution(prop_dict=dict, manager=manager)
            executor.start(blocking=True)
            CoreUtils.save_model(dict, filename)
            print("Saving and running again in non blocking mode")
            executor.start(blocking=False)
            while manager.running():
                executor.step()
            executor.finish()
            CoreUtils.save_model(dict, filename)


        :param blocking: Flag determining the execution mode, True by default
        :type blocking: bool
        """
        self._logger = debug_logger.DebugLogger(self._prop_dict)
        self._parent_control_barrier = SyncUtils.Barrier(1, timeout=0.0001, UseSpinLock=True)
        self._parent_proc = mp.Process(target=_supervisor_run, args=(self._prop_dict, self._parent_control_barrier))
        self._parent_proc.start()
        self._parent_control_barrier.parent_barrier()
        self._manager.start()
        self._finished=False
        self._stop_server = False
        self._t = threading.Thread(target=_run_debug_server, args=(self._prop_dict, self._port, lambda: self._stop_server))
        self._t.start()
        if blocking:
            while self._manager.running():
                self.step()
            self.finish()

    def step(self):
        """
        Executes a single step of simulation
        """
        # Quick turnover, data here is consistent
        self._logger.process_hookups()
        self._manager.fast_action()
        self._parent_control_barrier.resume_workers()
        # Workers execute
        # Longer turnover, data here might be inconsistent
        self._manager.slow_action()
        self._parent_control_barrier.parent_barrier(quit_workers=not self._manager.running())

    def finish(self):
        """
        After a session has been completed, finish removes all worker threads and closes debug server.
        """
        if not self._finished:
            logging.info("Attempting to stop threads")
            self._stop_server = True
            logging.info("Attempting to join parent")
            self._parent_proc.join()
            logging.info("attempting to join thread")
            self._t.join()
            self._manager.finish()
            self._finished=True


def run_model(prop_dict, manager, port=9000):
    """
    Simplified way of running a model.
    :param prop_dict: model dictionary
    :param manager: execution manager
    :param port: port for setting up a debugging server
    :return:
    """
    executor = ModelExecution(prop_dict=prop_dict, manager=manager, port=port)
    executor.start(blocking=True)
    executor.finish()
