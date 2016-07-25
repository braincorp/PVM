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
import argparse


class Manager(AbstractExecutionManager.ExecutionManager):
    def __init__(self, prop_dict, steps_to_run, cam):
        self.prop_dict = prop_dict
        self.steps_to_run = steps_to_run
        self._running = True
        self.cam = cam
        self.steps = 0

    def start(self):
        """
        This will be called right before the simulation starts
        """
        self.t_start = time.time()
        ret, frame = self.cam.read()
        self.yet_previous_frame1 = cv2.resize(frame, dsize=(self.prop_dict['current_array'].shape[1], self.prop_dict['current_array'].shape[0]))
        ret, frame = self.cam.read()
        self.yet_previous_frame = cv2.resize(frame, dsize=(self.prop_dict['current_array'].shape[1], self.prop_dict['current_array'].shape[0]))
        ret, frame = self.cam.read()
        self.previous_frame = cv2.resize(frame, dsize=(self.prop_dict['current_array'].shape[1], self.prop_dict['current_array'].shape[0]))
        ret, frame = self.cam.read()
        self.current_frame = cv2.resize(frame, dsize=(self.prop_dict['current_array'].shape[1], self.prop_dict['current_array'].shape[0]))

    def fast_action(self):
        """
        This is the time between steps of execution
        Data is consistent, but keep this piece absolutely minimal
        """
        # switch buffers
        self.prop_dict['yet_previous_array1'].copyto(self.yet_previous_frame1)
        self.prop_dict['yet_previous_array'].copyto(self.yet_previous_frame)
        self.prop_dict['previous_array'].copyto(self.previous_frame)
        self.prop_dict['current_array'].copyto(self.current_frame)
        pass

    def slow_action(self):
        """
        This is while the workers are running. You may do a lot of work here
        (preferably not more than the time of execution of workers).
        """
        ret, frame = self.cam.read()
        self.yet_previous_frame1 = self.yet_previous_frame
        self.yet_previous_frame = self.previous_frame
        self.previous_frame = self.current_frame
        self.current_frame = cv2.resize(frame, dsize=(self.prop_dict['current_array'].shape[1], self.prop_dict['current_array'].shape[0]))
        self.current_frame[0:30, 0:30, :] = np.random.randint(0, 255, (30, 30, 3))
        cv2.putText(self.current_frame, "X", (self.steps % 50 + 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.imshow("Current frame", self.prop_dict['current_array'].view(np.ndarray))
        cv2.imshow("Previous frame", self.prop_dict['previous_array'].view(np.ndarray))
        cv2.imshow("Predicted frame", self.prop_dict['predicted_array'].view(np.ndarray))
        cv2.imshow("First order error", self.prop_dict['first_order_error'].view(np.ndarray))
        cv2.imshow("Second order error", self.prop_dict['second_order_error'].view(np.ndarray))
        cv2.imshow("Difference", cv2.absdiff(self.prop_dict['predicted_array'].view(np.ndarray), self.prop_dict['previous_array'].view(np.ndarray)))
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
    simulation_dict['execution_unit_module'] = 'PVM_models.demo04_unit'
    unit = importlib.import_module(simulation_dict['execution_unit_module'])
    blocks_per_dim = 30
    block_size = 5
    simulation_dict['stage0_size'] = blocks_per_dim*blocks_per_dim
    yet_previous_array1 = SharedArray.SharedNumpyArray((block_size*blocks_per_dim, block_size*blocks_per_dim, 3), np.uint8)
    simulation_dict['yet_previous_array1'] = yet_previous_array1
    yet_previous_array = SharedArray.SharedNumpyArray((block_size*blocks_per_dim, block_size*blocks_per_dim, 3), np.uint8)
    simulation_dict['yet_previous_array'] = yet_previous_array
    previous_array = SharedArray.SharedNumpyArray((block_size*blocks_per_dim, block_size*blocks_per_dim, 3), np.uint8)
    simulation_dict['previous_array'] = previous_array
    current_array = SharedArray.SharedNumpyArray((block_size*blocks_per_dim, block_size*blocks_per_dim, 3), np.uint8)
    simulation_dict['current_array'] = current_array
    predicted_array = SharedArray.SharedNumpyArray((block_size*blocks_per_dim, block_size*blocks_per_dim, 3), np.uint8)
    simulation_dict['predicted_array'] = predicted_array
    first_order_error = SharedArray.SharedNumpyArray((block_size*blocks_per_dim, block_size*blocks_per_dim, 3), np.uint8)
    simulation_dict['first_order_error'] = first_order_error
    second_order_error = SharedArray.SharedNumpyArray((block_size*blocks_per_dim, block_size*blocks_per_dim, 3), np.uint8)
    simulation_dict['second_order_error'] = second_order_error
    learning_rate = SharedArray.SharedNumpyArray((1,), np.float)
    learning_rate[0] = 0.1
    simulation_dict['learning_rate'] = learning_rate
    momentum = SharedArray.SharedNumpyArray((1,), np.float)
    momentum[0] = 0.5
    simulation_dict['momentum'] = momentum
    for i in range(simulation_dict['stage0_size']):
        unit_parameters = {}
        unit_parameters['learning_rate'] = learning_rate
        unit_parameters['momentum'] = momentum
        unit_parameters['yet_previous_array1'] = yet_previous_array1
        unit_parameters['yet_previous_array'] = yet_previous_array
        unit_parameters['previous_array'] = previous_array
        unit_parameters['current_array'] = current_array
        unit_parameters['predicted_array'] = predicted_array
        unit_parameters['first_order_error'] = first_order_error
        unit_parameters['second_order_error'] = second_order_error
        # block will define a sub-block of the arena that each unit will be processing
        # it will contain (x0, y0, width, height)
        unit_parameters['block'] = SharedArray.SharedNumpyArray((4,), np.int)
        # each unit gets a 100x100 sub-block of the arena
        unit_parameters['block'][0] = (i / blocks_per_dim)*block_size
        unit_parameters['block'][1] = (i % blocks_per_dim)*block_size
        unit_parameters['block'][2] = block_size
        unit_parameters['block'][3] = block_size
        simulation_dict['stage0'].append(unit_parameters)
    for i in range(simulation_dict['stage0_size']):
        unit.ExecutionUnit.generate_missing_parameters(simulation_dict['stage0'][i])
    return simulation_dict


def run_demo(movie_file):
    """
    In this demo a future/predictive encoder is being instantiated to predict a camera image
    based on two previous frames. The encoder predicts the signal and its own error on that signal, that is
    additional set of units are trying to predict the magnitude of error between the prediction and the signal.
    In addition the hidden layer activations from the previous step of execution are used as the context block.
    Second order error is calculated as the error of the error prediction. Also, the learning rate of the primary signal
    is modulated by the magnitude of the second order error.
    """
    if movie_file != "":
        cam = cv2.VideoCapture(movie_file)
    else:
        cam = cv2.VideoCapture(-1)
    if not cam.isOpened():
        logging.error("Either cannot read the input file or no camera found!")
        exit(1)
    filename = "demo04_state.p.gz"
    if os.path.isfile(filename):
        state_dict = CoreUtils.load_model(filename)
    else:
        state_dict = generate_dict()
    manager = Manager(state_dict, 1000000, cam=cam)
    CoreUtils.run_model(state_dict, manager)
    CoreUtils.save_model(state_dict, filename)


if __name__ == '__main__':
    logging.basicConfig(filename="demo04.log", level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Send logs to stdout
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Movie file to load. If empty the script will try to use the camera", type=str, default="")
    args = parser.parse_args()
    run_demo(args.file)
