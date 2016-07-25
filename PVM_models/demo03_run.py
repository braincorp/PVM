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
        self.current_frame = None
        self.cam = cam

    def start(self):
        """
        This will be called right before the simulation starts
        """
        self.t_start = time.time()
        self.steps = 0
        ret, frame = self.cam.read()
        self.current_frame = cv2.resize(frame, dsize=(self.prop_dict['array'][0].shape[1], self.prop_dict['array'][0].shape[0]))
        self.current_frame1 = cv2.pyrDown(self.current_frame)
        self.current_frame2 = cv2.pyrDown(self.current_frame1)
        self.current_frame3 = cv2.resize(self.current_frame2, dsize=(self.prop_dict['array3'][0].shape[1], self.prop_dict['array3'][0].shape[0]))

    def fast_action(self):
        """
        This is the time between steps of execution
        Data is consistent, but keep this piece absolutely minimal
        """
        self.prop_dict['array'][:] = np.roll(self.prop_dict['array'], 1, axis=0)
        self.prop_dict['array'][0][:] = self.current_frame[:]
        self.prop_dict['array1'][:] = np.roll(self.prop_dict['array1'], 1, axis=0)
        self.prop_dict['array1'][0][:] = self.current_frame1[:]
        self.prop_dict['array2'][:] = np.roll(self.prop_dict['array2'], 1, axis=0)
        self.prop_dict['array2'][0][:] = self.current_frame2[:]
        self.prop_dict['array3'][:] = np.roll(self.prop_dict['array3'], 1, axis=0)
        self.prop_dict['array3'][0][:] = self.current_frame3[:]

    def slow_action(self):
        """
        This is while the workers are running. You may do a lot of work here
        (preferably not more than the time of execution of workers).
        """
        ret, frame = self.cam.read()
        self.current_frame = cv2.resize(frame, dsize=(self.prop_dict['array'][0].shape[1], self.prop_dict['array'][0].shape[0]))
        self.current_frame1 = cv2.pyrDown(self.current_frame)
        self.current_frame2 = cv2.pyrDown(self.current_frame1)
        self.current_frame3 = cv2.resize(self.current_frame2, dsize=(self.prop_dict['array3'][0].shape[1], self.prop_dict['array3'][0].shape[0]))

        im0 = np.hstack((self.prop_dict['array'][0].view(np.ndarray), self.prop_dict['predicted_array'][0].view(np.ndarray), cv2.absdiff(self.prop_dict['predicted_array'][0].view(np.ndarray), self.prop_dict['array'][0].view(np.ndarray))))
        im1 = np.hstack((self.prop_dict['array1'][0].view(np.ndarray), self.prop_dict['predicted_array1'][0].view(np.ndarray), cv2.absdiff(self.prop_dict['predicted_array1'][0].view(np.ndarray), self.prop_dict['array1'][0].view(np.ndarray))))
        im1 = cv2.resize(im1, dsize=(im0.shape[1], im0.shape[0]))
        im2 = np.hstack((self.prop_dict['array2'][0].view(np.ndarray), self.prop_dict['predicted_array2'][0].view(np.ndarray), cv2.absdiff(self.prop_dict['predicted_array2'][0].view(np.ndarray), self.prop_dict['array2'][0].view(np.ndarray))))
        im2 = cv2.resize(im2, dsize=(im0.shape[1], im0.shape[0]))
        im3 = np.hstack((self.prop_dict['array3'][0].view(np.ndarray), self.prop_dict['predicted_array3'][0].view(np.ndarray), cv2.absdiff(self.prop_dict['predicted_array3'][0].view(np.ndarray), self.prop_dict['array3'][0].view(np.ndarray))))
        im3 = cv2.resize(im3, dsize=(im0.shape[1], im0.shape[0]))
        im = np.vstack((im0, im1, im2, im3))
        cv2.imshow("Scale3", im)  # imshow is slow, its better to call it just once
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


def get_neighbours(x, y, sizex, sizey):
    result = []
    if x > 0:
        result.append([x-1, y])
    if y > 0:
        result.append([x, y-1])
    if x < sizex-1:
        result.append([x+1, y])
    if y < sizey-1:
        result.append([x, y+1])
    return result


def create_unit_parameters(learning_rate, momentum, block, array, predicted_array, internal_rep_size, context_block, output_block):
    unit_parameters = {}
    unit_parameters['learning_rate'] = learning_rate
    unit_parameters['momentum'] = momentum
    unit_parameters['array'] = array
    unit_parameters['predicted_array'] = predicted_array
    unit_parameters['internal_rep_size'] = internal_rep_size
    unit_parameters['output_block'] = output_block
    unit_parameters['context_blocks'] = [context_block]
    # block will define a sub-block of the arena that each unit will be processing
    # it will contain (x0, y0, width, height)
    unit_parameters['block'] = block
    return unit_parameters


def generate_dict(blocks_per_dim = 8, block_size = 8):
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
    simulation_dict['execution_unit_module'] = 'PVM_models.demo03_unit'
    unit = importlib.import_module(simulation_dict['execution_unit_module'])
    internal_rep_size = 2*block_size*block_size*3/10
    simulation_dict['stage0_size'] = 3*blocks_per_dim*blocks_per_dim+1

    # Create the high resolution x 3 frame array
    array = SharedArray.SharedNumpyArray((3, block_size*blocks_per_dim, block_size*blocks_per_dim, 3), np.uint8)
    simulation_dict['array'] = array
    predicted_array = SharedArray.SharedNumpyArray((1, block_size*blocks_per_dim, block_size*blocks_per_dim, 3), np.uint8)
    simulation_dict['predicted_array'] = predicted_array

    # Create the medium resolution x 12 frame array
    array1 = SharedArray.SharedNumpyArray((4*3, block_size*blocks_per_dim/2, block_size*blocks_per_dim/2, 3), np.uint8)
    simulation_dict['array1'] = array1
    predicted_array1 = SharedArray.SharedNumpyArray((4, block_size*blocks_per_dim/2, block_size*blocks_per_dim/2, 3), np.uint8)
    simulation_dict['predicted_array1'] = predicted_array1

    # Create the low resolution x 48 frame array
    array2 = SharedArray.SharedNumpyArray((16*3, block_size*blocks_per_dim/4, block_size*blocks_per_dim/4, 3), np.uint8)
    simulation_dict['array2'] = array2
    predicted_array2 = SharedArray.SharedNumpyArray((16, block_size*blocks_per_dim/4, block_size*blocks_per_dim/4, 3), np.uint8)
    simulation_dict['predicted_array2'] = predicted_array2

    # Create the low resolution x 3 frame array
    array3 = SharedArray.SharedNumpyArray((3, block_size, block_size, 3), np.uint8)
    simulation_dict['array3'] = array3
    predicted_array3 = SharedArray.SharedNumpyArray((1, block_size, block_size, 3), np.uint8)
    simulation_dict['predicted_array3'] = predicted_array3

    context_block = SharedArray.SharedNumpyArray((internal_rep_size,), np.float)

    # Create some basic MLP parameters
    learning_rate = SharedArray.SharedNumpyArray((1,), np.float)
    learning_rate[0] = 0.01
    simulation_dict['learning_rate'] = learning_rate
    momentum = SharedArray.SharedNumpyArray((1,), np.float)
    momentum[0] = 0.5
    simulation_dict['momentum'] = momentum

    # Run loops to create and connect all the units
    # High spatial resolution, low temporal cover layer
    for x in range(blocks_per_dim):
        for y in range(blocks_per_dim):
            output_block = SharedArray.SharedNumpyArray((internal_rep_size,), np.float)
            block = SharedArray.SharedNumpyArray((4,), np.int)
            block[0] = x*block_size
            block[1] = y*block_size
            block[2] = block_size
            block[3] = block_size
            unit_parameters = create_unit_parameters(learning_rate,
                                                     momentum,
                                                     block,
                                                     array,
                                                     predicted_array,
                                                     internal_rep_size,
                                                     context_block,
                                                     output_block)
            simulation_dict['stage0'].append(unit_parameters)

    # Medium spatial resolution, medium temporal cover layer
    for x in range(blocks_per_dim):
        for y in range(blocks_per_dim):
            output_block = SharedArray.SharedNumpyArray((internal_rep_size,), np.float)
            block = SharedArray.SharedNumpyArray((4,), np.int)
            block[0] = x*(block_size/2)
            block[1] = y*(block_size/2)
            block[2] = block_size/2
            block[3] = block_size/2
            unit_parameters = create_unit_parameters(learning_rate,
                                                     momentum,
                                                     block,
                                                     array1,
                                                     predicted_array1,
                                                     internal_rep_size,
                                                     context_block,
                                                     output_block)
            simulation_dict['stage0'].append(unit_parameters)

    # Low spatial resolution, high temporal cover layer
    for x in range(blocks_per_dim):
        for y in range(blocks_per_dim):
            output_block = SharedArray.SharedNumpyArray((internal_rep_size,), np.float)
            block = SharedArray.SharedNumpyArray((4,), np.int)
            block[0] = x*(block_size/4)
            block[1] = y*(block_size/4)
            block[2] = block_size/4
            block[3] = block_size/4
            unit_parameters = create_unit_parameters(learning_rate,
                                                     momentum,
                                                     block,
                                                     array2,
                                                     predicted_array2,
                                                     internal_rep_size,
                                                     context_block,
                                                     output_block)
            simulation_dict['stage0'].append(unit_parameters)

    # Each upper layer will provide its internal activations as context to the lower layer.
    for x in range(blocks_per_dim):
        for y in range(blocks_per_dim):
            i = x*blocks_per_dim+y
            simulation_dict['stage0'][i+blocks_per_dim*blocks_per_dim]['context_blocks'].append(simulation_dict['stage0'][i+2*blocks_per_dim*blocks_per_dim]['output_block'])
            simulation_dict['stage0'][i]['context_blocks'].append(simulation_dict['stage0'][i+blocks_per_dim*blocks_per_dim]['output_block'])
            surround = get_neighbours(x, y, blocks_per_dim, blocks_per_dim)
            surround_idx = map(lambda a: a[0]*blocks_per_dim+a[1], surround)
            for idx in surround_idx:
                simulation_dict['stage0'][i]['context_blocks'].append(simulation_dict['stage0'][idx]['output_block'])
                simulation_dict['stage0'][i+blocks_per_dim*blocks_per_dim]['context_blocks'].append(simulation_dict['stage0'][idx+blocks_per_dim*blocks_per_dim]['output_block'])
                simulation_dict['stage0'][i+2*blocks_per_dim*blocks_per_dim]['context_blocks'].append(simulation_dict['stage0'][idx+2*blocks_per_dim*blocks_per_dim]['output_block'])

    # This single unit will cover the entire scene and supply its internal activations as
    # context to everyone.
    for i in range(1):
        output_block = context_block
        block = SharedArray.SharedNumpyArray((4,), np.int)
        # each unit gets a 100x100 sub-block of the arena
        block[0] = 0
        block[1] = 0
        block[2] = block_size
        block[3] = block_size
        unit_parameters = create_unit_parameters(learning_rate,
                                                 momentum,
                                                 block,
                                                 array3,
                                                 predicted_array3,
                                                 internal_rep_size,
                                                 context_block,
                                                 output_block)
        simulation_dict['stage0'].append(unit_parameters)

    for i in range(simulation_dict['stage0_size']):
        unit.ExecutionUnit.generate_missing_parameters(simulation_dict['stage0'][i])
    return simulation_dict


def run_demo(movie_file):
    """
    In this demo, an image is being predicted by a set of predictive encoders looking
    at different aspects of the input. There is a set of future encoders digesting two 8x8 frames to predict
    the next one, another set taking 8 4x4 frames to predict 4 subsequent frames and another set taking
    32 2x2 frames to predict 16 subsequent frames. Additionally there is one unit taking the whole image as
    8x8 block whose internal representations are shared as context with all the other units.
    The system has feedback connections, more temporal areas feed back to more spatial areas. Also cross-like
    neighbourhood of lateral projections is instantiated.


    The simulation is synchronous, runs in single stage.
    """
    filename = "demo03_state.p.gz"
    if movie_file != "":
        cam = cv2.VideoCapture(movie_file)
    else:
        cam = cv2.VideoCapture(-1)
    if not cam.isOpened():
        logging.error("Either cannot read the input file or no camera found!")
        exit(1)
    if os.path.isfile(filename):
        state_dict = CoreUtils.load_model(filename)
    else:
        state_dict = generate_dict()
    manager = Manager(state_dict, 1000000, cam=cam)
    CoreUtils.run_model(state_dict, manager)
    CoreUtils.save_model(state_dict, filename)


if __name__ == '__main__':
    logging.basicConfig(filename="demo03.log", level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Send logs to stdout
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Movie file to load. If empty the script will try to use the camera", type=str, default="")
    args = parser.parse_args()
    run_demo(args.file)
