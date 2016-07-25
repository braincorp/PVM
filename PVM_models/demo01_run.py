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
import PVM_framework.CoreUtils as CoreUtils
import PVM_framework.PVM_Create as PVM_Create
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
        image = (self.prop_dict['arena']+1).astype(np.uint8)*127
        cv2.imshow("arena", image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or self.steps > self.steps_to_run:
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
    simulation_dict['execution_unit_module'] = 'PVM_models.demo01_unit'
    unit = importlib.import_module(simulation_dict['execution_unit_module'])
    simulation_dict['stage0_size'] = 100
    arena = SharedArray.SharedNumpyArray((1000, 1000), np.int8)  # the arena will be 1000x1000 pixels in this case
    arena.copyto(2*np.random.randint(2, size=arena.shape).astype(np.int8)-1)
    simulation_dict['arena'] = arena
    temperature = SharedArray.SharedNumpyArray((1,), np.float)
    temperature[0] = 1/2.269
    simulation_dict['temperature'] = temperature
    for i in range(simulation_dict['stage0_size']):
        unit_parameters = {}
        unit_parameters['arena'] = arena
        unit_parameters['temperature'] = temperature
        # block will define a sub-block of the arena that each unit will be processing
        # it will contain (x0, y0, width, height)
        unit_parameters['block'] = SharedArray.SharedNumpyArray((4,), np.int)
        # each unit gets a 100x100 sub-block of the arena
        unit_parameters['block'][0] = (i / 10)*100
        unit_parameters['block'][1] = (i % 10)*100
        unit_parameters['block'][2] = 100
        unit_parameters['block'][3] = 100
        simulation_dict['stage0'].append(unit_parameters)
    for i in range(simulation_dict['stage0_size']):
        unit.ExecutionUnit.generate_missing_parameters(simulation_dict['stage0'][i])
    return simulation_dict


def run_demo():
    """
    In this simple demo the crticial temperature Ising model is run synchronously by a set of units on a large 1000x1000
    domain. To make things fast the worker code if written in cython.
    :return:
    """
    filename = "demo01_state.p.gz"
    if os.path.isfile(filename):
        sate_dict = CoreUtils.load_model(filename)
    else:
        sate_dict = generate_dict()
    manager = Manager(sate_dict, 1000000)
    CoreUtils.run_model(sate_dict, manager)
    CoreUtils.save_model(sate_dict, filename)


if __name__ == '__main__':
    logging.basicConfig(filename="demo01.log", level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Send logs to stdout
    run_demo()
