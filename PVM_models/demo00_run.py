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

import multiprocessing as mp
import time
import cv2
import numpy as np
import PVM_framework.PVM_Create as PVM_Create
import PVM_framework.CoreUtils as CoreUtils
import PVM_framework.SharedArray as SharedArray
import PVM_framework.AbstractExecutionManager as AbstractExecutionManager
import os
import logging
import importlib


class Manager(AbstractExecutionManager.ExecutionManager):
    def __init__(self, prop_dict, steps_to_run):
        self.prop_dict = prop_dict
        self.steps_to_run = steps_to_run
        self._running = True

    def start(self):
        """
        This will be called right before the simulation starts
        """
        self.t_start = time.time()
        self._running = True
        self.steps = 0

    def fast_action(self):
        """
        This is the time between steps of execution
        Data is consistent, but keep this piece absolutely minimal
        """
        pass

    def slow_action(self):
        """
        This is while the workers are running. You may do a lot of work here
        (preferably not more than the time of execution of workers).
        """
        cv2.imshow("arena", self.prop_dict['arena'].view(np.ndarray))
        cv2.waitKey(1)
        if self.steps > self.steps_to_run:
            self._running = False
        now = time.time()
        self.steps += 1
        sps = self.steps / (now-self.t_start)
        print ("%3.3f steps per sec" % (sps)) + "\r",

    def running(self):
        """
        While returning True the simulation will keep going.
        """
        return self._running

    def finish(self):
        pass


def generate_dict():
    # simulation dictionary has to have:
    # N - SharedArray of one entity enumerating the step of the simulation
    # stages - integer number of processing stages
    # num_proc - integer number of processors that will be utilized for workers
    # stage0 - a list of element parameters (dictionaries) for each execution element
    # stage0_size - number of execution elements in the stage
    # other parameters are model specific.
    simulation_dict = PVM_Create.create_blank_dictionary()
    simulation_dict['stages'] = 1
    simulation_dict['num_proc'] = mp.cpu_count()/2
    simulation_dict['stage0'] = []
    simulation_dict['execution_unit_module'] = 'PVM_models.demo00_unit'
    unit = importlib.import_module(simulation_dict['execution_unit_module'])
    simulation_dict['stage0_size'] = 100
    arena = SharedArray.SharedNumpyArray((500, 500), np.uint8)  # the arena will be 500x500 in this case
    simulation_dict['arena'] = arena
    for i in range(simulation_dict['stage0_size']):
        unit_parameters = {}
        unit_parameters['arena'] = arena
        # block will define a sub-block of the arena that each unit will be processing
        # it will contain (x0, y0, width, height)
        unit_parameters['block'] = SharedArray.SharedNumpyArray((4,), np.int)
        # each unit gets a 100x100 sub-block of the arena
        unit_parameters['block'][0] = (i / 10)*50
        unit_parameters['block'][1] = (i % 10)*50
        unit_parameters['block'][2] = 50
        unit_parameters['block'][3] = 50
        simulation_dict['stage0'].append(unit_parameters)
    for i in range(simulation_dict['stage0_size']):
        unit.ExecutionUnit.generate_missing_parameters(simulation_dict['stage0'][i])
    return simulation_dict


def run_demo():
    """
    In this very simple demo a set of workers operate on a 500x500 image domain by randomly flipping
    selected bits. To make things faster the bit/byte flipping function is written in cython.
    :return:
    """
    filename = "demo00_state.p.gz"
    if os.path.isfile(filename):
        state_dict = CoreUtils.load_model(filename)
    else:
        state_dict = generate_dict()
    manager = Manager(state_dict, 1000)
    executor = CoreUtils.ModelExecution(prop_dict=state_dict, manager=manager)
    executor.start(blocking=True)
    CoreUtils.save_model(state_dict, filename)
    print("Saving and running again in non blocking mode")
    executor.start(blocking=False)
    while manager.running():
        executor.step()
    executor.finish()
    CoreUtils.save_model(state_dict, filename)


if __name__ == '__main__':
    logging.basicConfig(filename="demo00.log", level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')
    run_demo()
