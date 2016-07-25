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
import numpy as np
import PVM_framework.SharedArray as SharedArray
import logging
import importlib
import random
import datetime
import glob

PVM_FLAG_CTRL_DREAM = 0
PVM_FLAG_CTRL_DREAM_EXP = 1
PVM_FLAG_CTRL_LEARNING = 2

PVM_FLAG_VAL_TRIGGER = 1
PVM_FLAG_VAL_CANCEL = 2
PVM_FLAG_VAL_RESET = 0
PVM_FLAG_TRIGGER_DISPLAY = 15
PVM_FLAG_TRIGGER_RECORD = 16
PVM_MAX_LAYERS = 20  # Arbitrarily set to 20, could be anything really
PVM_LOG_ERROR_EVERY = 1000  # Interval for error logging

PVM_FLAG_VAL_DREAM = 1
PVM_FLAG_VAL_BLINDSPOT = 2
PVM_FLAG_VAL_GRAY = 3
PVM_FLAG_VAL_NOISE = 4
PVM_FLAG_VAL_NOISE_SPOT = 5
PVM_FLAG_VAL_INV_BLINDSPOT = 6
PVM_FLAG_VAL_INV_NOISE_SPOT = 10
PVM_FLAG_VAL_DREAM_SPOT = 11
PVM_FLAG_VAL_INV_DREAM_SPOT = 7
PVM_FLAG_VAL_DEEP_DREAM = 12
PVM_FLAG_VAL_BLINKS = 8
PVM_FLAG_VAL_NOISY_SIGNAL = 9

PVM_LEARNING_FLAG = 2
PVM_LEARNING_RESET = 0
PVM_LEARNING_FREEZE = 1
PVM_LEARNING_UNFREEZE = 2

PVM_PAUSE = 1
PVM_RESUME = 0
PVM_DUMP = 2


def create_blank_dictionary(name="default", description="default", save_sources=False):
    """
    Creates a minimal model dictionary.

    :return: model dict
    :rtype: dict
    """
    import PVM_framework.SharedArray as SharedArray
    import numpy as np
    simulation_dict = {}
    simulation_dict['N'] = SharedArray.SharedNumpyArray((1,), np.int64)
    simulation_dict['N'][0] = 0
    simulation_dict['record_filename'] = SharedArray.SharedNumpyArray((256,), np.uint8)
    simulation_dict['paused'] = SharedArray.SharedNumpyArray((1,), np.int)
    simulation_dict['paused'][0] = 0
    simulation_dict['finished'] = SharedArray.SharedNumpyArray((1,), np.int)
    simulation_dict['finished'][0] = 0
    simulation_dict['flags'] = SharedArray.SharedNumpyArray((16,), np.uint8)
    for i in range(16):
        simulation_dict['flags'][i] = 0
    simulation_dict['num_proc'] = mp.cpu_count()/2
    simulation_dict['debug_infrastructure'] = {}
    simulation_dict['debug_infrastructure']['enabled_hookups'] = {}
    simulation_dict['debug_infrastructure']['disabled_hookups'] = {}
    logging.info("Created a blank dictionary")
    simulation_dict['description'] = description
    simulation_dict['name'] = name
    simulation_dict['sources'] = {}
    if save_sources:
        for filename in glob.glob("*.py"):
            f = open(filename, "r")
            simulation_dict['sources'][filename] = f.read()
            f.close()
            logging.info("Saved source file %s into the dictionary" % filename)
        for filename in glob.glob("*.pyx"):
            f = open(filename, "r")
            simulation_dict['sources'][filename] = f.read()
            f.close()
            logging.info("Saved source file %s into the dictionary" % filename)
    simulation_dict['hash'] = "%08x" % random.getrandbits(32)
    simulation_dict['timestamp'] = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    logging.info("Assigned hash %s to this simulation instance" % simulation_dict['hash'])
    logging.info("Assigned timestamp %s to this simulation instance" % simulation_dict['timestamp'])
    return simulation_dict


def get_surround(xy, dim_x=10, dim_y=10, radius=1, exclude_self=True):
    """
    Returns the indices of elements on the grid that are within square radius
    of the given xy

      radius = 1:

        0 1 0
        1 1 1
        0 1 0

      radius = 1.5:

        1 1 1
        1 1 1
        1 1 1

      radius = 2

        0 0 1 0 0
        0 1 1 1 0
        1 1 1 1 1
        0 1 1 1 0
        0 0 1 0 0

    Setting exclude_self to True removes the center unit
    """
    laterals = []
    for dx in range(-int(radius), int(radius)+1, 1):
        for dy in range(-int(radius), int(radius)+1, 1):
            if dx**2 + dy**2 > radius**2:
                continue
            if (xy[0]+dx >= 0) and (xy[0]+dx < dim_x) and (xy[1]+dy >= 0) and (xy[1]+dy < dim_y):
                if not (exclude_self and dx == 0 and dy == 0):
                    laterals.append((xy[0]+dx, xy[1]+dy))
    return laterals


def get_fan_in(xy=(0, 0), dim_x_l=10, dim_y_l=10, dim_x_u=9, dim_y_u=9, block_x=2, block_y=2, radius=2):
    """
    Selects a block_x x block_y subsquare in the underlying layers lying directly below the unit in the
    upper layer. Selects units within radius in that block.

      e.g. block_x=2, block_y=2 radius=2

        1 1
        1 1

      e.g. block_x=3, block_y=3 radius=2

        1 1 1
        1 1 1
        1 1 1

      e.g. block_x=3, block_y=3 radius=1

        0 1 0
        1 1 1
        0 1 0

    """
    x = xy[0]
    y = xy[1]
    if dim_x_u > 1:
        factor_x = ((dim_x_l-1)-(block_x-1))/(1.0*(dim_x_u-1))
    else:
        factor_x = ((dim_x_l-1)-(block_x))/2.0
    if dim_y_u > 1:
        factor_y = ((dim_y_l-1)-(block_y-1))/(1.0*(dim_y_u-1))
    else:
        factor_y = ((dim_y_l-1)-(block_y))/2.0
    results = []
    if dim_x_u > 1 and dim_y_u > 1:
        for xx in range(block_x):
            for yy in range(block_y):
                if (xx-(block_x-1)*0.5)**2 + (yy-(block_y-1)*0.5)**2 > radius**2:
                    continue
                results.append((int((factor_x*(x))+xx), int((factor_y*(y))+yy)))
        return results
    elif dim_x_u == 1 and dim_y_u > 1:
        for xx in range(block_x):
            for yy in range(block_y):
                if (xx-(block_x-1)*0.5)**2 + (yy-(block_y-1)*0.5)**2 > radius**2:
                    continue
                results.append((int((dim_x_l-block_x)/2.0+xx), int((factor_y*(y)+yy))))
        return results
    elif dim_x_u > 1 and dim_y_u == 1:
        for xx in range(block_x):
            for yy in range(block_y):
                if (xx-(block_x-1)*0.5)**2 + (yy-(block_y-1)*0.5)**2 > radius**2:
                    continue
                results.append((int((factor_x*(x)+xx)), int((dim_y_l-block_y)/2.0+yy)))
        return results
    elif dim_x_u == 1 and dim_y_u == 1:
        for xx in range(block_x):
            for yy in range(block_y):
                if (xx-(block_x-1)*0.5)**2 + (yy-(block_y-1)*0.5)**2 > radius**2:
                    continue
                results.append((int((dim_x_l-block_x)/2.0+xx), int((dim_y_l-block_y)/2.0+yy)))
        return results


def connect_forward_and_back(simulation_dict, (index0, blocks_per_dim0, predicted_array), (index1, blocks_per_dim1), square_size, radius, context_factor):
    """
    Connect two layers with a given fan-in as defined by the square_size and radius.
    The forward connections are accompanied by a feedback context connections back to
    the originating source unit.

    :param simulation_dict:
    :param square_size:
    :param radius:
    :param context_factor:
    :return:
    """
    hidden_size = simulation_dict['hidden_size']
    dx = hidden_size
    dy = hidden_size
    logging.info("Connecting from index %d to index %d" % (index0, index1))
    logging.info("Input layer size is %d, receiving layer size is %d" % (blocks_per_dim0, blocks_per_dim1))
    logging.info("Radius of connectivity %d" % radius)
    for x in range(blocks_per_dim1):
        for y in range(blocks_per_dim1):
            surround = get_fan_in((x, y),
                                  dim_x_l=blocks_per_dim0,
                                  dim_y_l=blocks_per_dim0,
                                  dim_x_u=blocks_per_dim1,
                                  dim_y_u=blocks_per_dim1,
                                  block_x=square_size,
                                  block_y=square_size,
                                  radius=radius)
            dest = index1 + x * (blocks_per_dim1) + y  # destination unit
            for xy in surround:
                source = index0 + xy[0] * blocks_per_dim0 + xy[1]  # source unit
                # Prepare the input and corresponding delta block at source
                input_block = simulation_dict['stage0'][source]['output_block']
                delta_block = SharedArray.SharedNumpyArray_like(input_block)
                simulation_dict['stage0'][source]['delta_blocks'].append(delta_block)
                # Prepare the context and corresonding delta block at destination
                context_block = simulation_dict['stage0'][dest]['output_block']
                delta_block2 = SharedArray.SharedNumpyArray_like(context_block)
                simulation_dict['stage0'][dest]['delta_blocks'].append(delta_block2)
                # Connect the context block to the source
                simulation_dict['stage0'][source]['context_blocks'].append((context_block, delta_block2, context_factor))
                # Prepare the predicted blocks
                xx = xy[0]*hidden_size
                yy = xy[1]*hidden_size
                assert(predicted_array[xx:xx+dx, yy:yy+dy].shape == context_block.shape)
                predicted_block = SharedArray.DynamicView(predicted_array)[xx:xx+dx, yy:yy+dy]
                if not (predicted_block.shape == (dx, dy)):
                    print predicted_block.shape
                    raise
                # Connect the input to the destination together with its predicted blocks and so on.
                past_block = SharedArray.SharedNumpyArray_like(input_block)
                derivative_block = SharedArray.SharedNumpyArray_like(input_block)
                integral_block = SharedArray.SharedNumpyArray_like(input_block)
                pred_block_local = SharedArray.SharedNumpyArray_like(input_block)
                simulation_dict['stage0'][dest]['signal_blocks'].append((input_block, delta_block, predicted_block, past_block, derivative_block, integral_block, pred_block_local))


def connect_forward_and_back_v1(simulation_dict, (index0, blocks_per_dim0, predicted_array, predicted_array_t2), (index1, blocks_per_dim1), square_size, radius, context_factor):
    """
    Connect two layers with a given fan in as defined by the square_size and radius.
    The forward connections are accompanied by feedback context connections back to
    the originating source unit.

    :param simulation_dict:
    :param square_size:
    :param radius:
    :param context_factor:
    :return:
    """
    hidden_size = simulation_dict['hidden_size']
    dx = hidden_size
    dy = hidden_size
    logging.info("Connecting from index %d to index %d" % (index0, index1))
    logging.info("Input layer size is %d, receiving layer size is %d" % (blocks_per_dim0, blocks_per_dim1))
    logging.info("Radius of connectivity %d" % radius)
    for x in range(blocks_per_dim1):
        for y in range(blocks_per_dim1):
            surround = get_fan_in((x, y),
                                  dim_x_l=blocks_per_dim0,
                                  dim_y_l=blocks_per_dim0,
                                  dim_x_u=blocks_per_dim1,
                                  dim_y_u=blocks_per_dim1,
                                  block_x=square_size,
                                  block_y=square_size,
                                  radius=radius)
            dest = index1 + x * (blocks_per_dim1) + y  # destination unit
            for xy in surround:
                source = index0 + xy[0] * blocks_per_dim0 + xy[1]  # source unit
                # Prepare the input and corresponding delta block at source
                input_block = simulation_dict['stage0'][source]['output_block']
                delta_block = SharedArray.SharedNumpyArray_like(input_block)
                simulation_dict['stage0'][source]['delta_blocks'].append(delta_block)
                # Prepare the context and corresonding delta block at destination
                context_block = simulation_dict['stage0'][dest]['output_block']
                delta_block2 = SharedArray.SharedNumpyArray_like(context_block)
                simulation_dict['stage0'][dest]['delta_blocks'].append(delta_block2)
                # Connect the context block to the source
                simulation_dict['stage0'][source]['context_blocks'].append((context_block, delta_block2, context_factor))
                # Prepare the predicted blocks
                xx = xy[0]*hidden_size
                yy = xy[1]*hidden_size
                assert(predicted_array[xx:xx+dx, yy:yy+dy].shape == context_block.shape)
                predicted_block = SharedArray.DynamicView(predicted_array)[xx:xx+dx, yy:yy+dy]
                predicted_block2 = SharedArray.DynamicView(predicted_array_t2)[xx:xx+dx, yy:yy+dy]
                if not (predicted_block.shape == (dx, dy)):
                    print predicted_block.shape
                    raise
                # Connect the input to the destination together with its predicted blocks and so on.
                simulation_dict['stage0'][dest]['signal_blocks'].append((input_block, delta_block, predicted_block, predicted_block2))


def connect_back(simulation_dict, (index_from, blocks_per_dim_from), (index_to, blocks_per_dim_to), square_size, radius, context_factor):
    """
    Connect feedback only. This function is used to connect two layers that are not directly connected (in which
    case the feedback connection would have been established along with the feedforward connection), e.g. in cases
    when feedback is sent from some layer way above to some lower layer.

    :param simulation_dict:
    :param square_size:
    :param radius:
    :param context_factor:
    :return:
    """
    logging.info("Connecting back additional context from index %d to index %d" % (index_from, index_to))
    logging.info("Connecting back additional context from layer size is %d, receiving layer size is %d" % (blocks_per_dim_from, blocks_per_dim_to))
    logging.info("Radius of connectivity %d" % radius)
    for x in range(blocks_per_dim_from):
        for y in range(blocks_per_dim_from):
            surround = get_fan_in((x, y),
                                  dim_x_l=blocks_per_dim_to,
                                  dim_y_l=blocks_per_dim_to,
                                  dim_x_u=blocks_per_dim_from,
                                  dim_y_u=blocks_per_dim_from,
                                  block_x=square_size,
                                  block_y=square_size,
                                  radius=radius)
            source = index_from + x * (blocks_per_dim_from) + y  # unit in the higher layer
            for xy in surround:
                dest = index_to + xy[0] * blocks_per_dim_to + xy[1]  # unit in the lower layer
                context_block = simulation_dict['stage0'][source]['output_block']
                delta_block2 = SharedArray.SharedNumpyArray_like(context_block)
                simulation_dict['stage0'][source]['delta_blocks'].append(delta_block2)
                # Connect the context block to the source
                simulation_dict['stage0'][dest]['context_blocks'].append((context_block, delta_block2, context_factor))


def gather_surround(simulation_dict, (index0, blocks_per_dim0), radius, context_factor, exclude_self=True):
    for x in range(blocks_per_dim0):
        for y in range(blocks_per_dim0):
            surround = get_surround((x, y), dim_x=blocks_per_dim0, dim_y=blocks_per_dim0, radius=radius, exclude_self=exclude_self)
            dest = index0 + x * blocks_per_dim0 + y  # destination unit
            for xy in surround:
                source = xy[0] * blocks_per_dim0 + xy[1]  # source unit
                context_block = simulation_dict['stage0'][source]['output_block']
                delta_block = SharedArray.SharedNumpyArray_like(context_block)
                simulation_dict['stage0'][source]['delta_blocks'].append(delta_block)
                simulation_dict['stage0'][dest]['context_blocks'].append((context_block, delta_block, context_factor))


def create_basic_unit_v1(learning_rate, momentum, tau, readout_learning_rate):
    unit_parameters = dict()
    unit_parameters['tau'] = tau
    unit_parameters['primary_learning_rate'] = learning_rate
    unit_parameters['readout_learning_rate'] = readout_learning_rate
    unit_parameters['momentum'] = momentum
    unit_parameters['signal_blocks'] = []
    unit_parameters['readout_blocks'] = []
    unit_parameters['predicted_blocks'] = []
    unit_parameters['delta_blocks'] = []
    unit_parameters['context_blocks'] = []
    return unit_parameters


def generate_dict_options(name, description, options):
    if options['version_major'] == "1" and options["version_minor"] == "0":
        dic_ = generate_v1(name=name,
                           description=description,
                           options=options
                           )
        return dic_


def generate_v1(name, description, options):
    input_block_size = int(options["input_block_size"])
    hidden_size = int(options["hidden_block_size"])
    layer_shape = map(lambda x: int(x), options["layer_shapes"])
    readout_block_size = map(lambda x: int(x), options["readout_block_size"])
    readout_layer = map(lambda x: x == "1", options["enable_readout"])
    lateral_radius = float(options["lateral_radius"])
    fan_in_square_size = int(options["fan_in_square_size"])
    fan_in_radius = int(options["fan_in_radius"])
    readout_depth = int(options["readout_depth"])
    ex_module = options["ex_module"]
    exclude_self = (options["context_exclude_self"] == '1')
    last_layer_context_to_all = (options["last_layer_context_to_all"] == '1')
    send_context_two_layers_back = (options["send_context_two_layers_back"] == '1')
    simulation_dict = create_blank_dictionary(name=name, description=description, save_sources=(options["save_source_files"] == '1'))
    simulation_dict['stages'] = 1
    simulation_dict['num_proc'] = 2*mp.cpu_count()/3
    simulation_dict['stage0'] = []
    simulation_dict['execution_unit_module'] = ex_module
    simulation_dict['version_major'] = 1
    simulation_dict['version_minor'] = 0
    unit = importlib.import_module(simulation_dict['execution_unit_module'])
    blocks_per_dim = layer_shape
    layers = len(blocks_per_dim)
    error_log = SharedArray.SharedNumpyArray((layers+1, 1000000), np.float)
    simulation_dict['error_log'] = error_log

    simulation_dict['input_block_size'] = input_block_size
    simulation_dict['hidden_size'] = hidden_size
    simulation_dict['learning_rates'] = []
    simulation_dict['momenta'] = []
    simulation_dict['taus'] = []
    simulation_dict['predicted_arrays'] = []
    simulation_dict['predicted_arrays_t2'] = []
    simulation_dict['predicted_readout_arrays'] = []
    simulation_dict['readout_arrays'] = []
    simulation_dict['state_arrays'] = []
    simulation_dict['delta_arrays'] = []

    input_array = SharedArray.SharedNumpyArray((input_block_size*blocks_per_dim[0], input_block_size*blocks_per_dim[0], 3), np.uint8)
    simulation_dict['input_array'] = input_array
    input_array_float = SharedArray.SharedNumpyArray((input_block_size*blocks_per_dim[0], input_block_size * blocks_per_dim[0], 3), np.float)
    simulation_dict['input_array_float'] = input_array_float

    for (i, bpd) in enumerate(blocks_per_dim):
        if readout_layer[i]:
            readout_array_float00 = SharedArray.SharedNumpyArray((bpd*readout_block_size[i], bpd*readout_block_size[i], readout_depth), np.float)
            simulation_dict['readout_arrays'].append(readout_array_float00)
            predicted_readout_array_float00 = SharedArray.SharedNumpyArray((bpd*readout_block_size[i], bpd*readout_block_size[i], readout_depth), np.float)
            simulation_dict['predicted_readout_arrays'].append(predicted_readout_array_float00)

    # input array 0 is a special case, 3 dimensions because of color input
    predicted_array0 = SharedArray.SharedNumpyArray((input_block_size*blocks_per_dim[0], input_block_size*blocks_per_dim[0], 3), np.float)
    simulation_dict['predicted_arrays'].append(predicted_array0)
    predicted_array2 = SharedArray.SharedNumpyArray((input_block_size*blocks_per_dim[0], input_block_size*blocks_per_dim[0], 3), np.float)
    simulation_dict['predicted_arrays_t2'].append(predicted_array2)
    delta_array0 = SharedArray.SharedNumpyArray((input_block_size*blocks_per_dim[0], input_block_size*blocks_per_dim[0], 3), np.float)
    simulation_dict['delta_arrays'].append(delta_array0)
    # All the rest is generic
    for (i, bpd) in enumerate(blocks_per_dim[:-1]):
        predicted_array1 = SharedArray.SharedNumpyArray((hidden_size*bpd, hidden_size*bpd), np.float)
        simulation_dict['predicted_arrays'].append(predicted_array1)
        predicted_array2 = SharedArray.SharedNumpyArray((hidden_size*bpd, hidden_size*bpd), np.float)
        simulation_dict['predicted_arrays_t2'].append(predicted_array2)
        delta_array1 = SharedArray.SharedNumpyArray((hidden_size*bpd, hidden_size*bpd), np.float)
        simulation_dict['delta_arrays'].append(delta_array1)
    for (i, bpd) in enumerate(blocks_per_dim):
        state_array0 = SharedArray.SharedNumpyArray((hidden_size*bpd, hidden_size*bpd), np.float)
        simulation_dict['state_arrays'].append(state_array0)

    # Base learning rate
    for (i, bpd) in enumerate(blocks_per_dim):
        learning_rate = SharedArray.SharedNumpyArray((1, ), np.float)
        learning_rate[0] = 0.0
        simulation_dict['learning_rates'].append(learning_rate)
    additional_learning_rate = SharedArray.SharedNumpyArray((1, ), np.float)
    additional_learning_rate[0] = 0.0
    simulation_dict['readout_learning_rate'] = additional_learning_rate
    # Momentum is the same everywhere.
    momentum = SharedArray.SharedNumpyArray((1, ), np.float)
    momentum[0] = float(options["momentum"])
    simulation_dict['momentum'] = momentum
    # Tau is the integration constant for the signal integral
    tau = SharedArray.SharedNumpyArray((1, ), np.float)
    tau[0] = float(options['tau'])
    simulation_dict['tau'] = tau
    context_factor_lateral = SharedArray.SharedNumpyArray((1, ), np.float)
    context_factor_lateral[0] = 0.0
    simulation_dict['context_factor_lateral'] = context_factor_lateral
    context_factor_feedback = SharedArray.SharedNumpyArray((1, ), np.float)
    context_factor_feedback[0] = 0.0
    simulation_dict['context_factor_feedback'] = context_factor_feedback
    base_index = [0]
    for bpd in blocks_per_dim:
        base_index.append(base_index[-1] + bpd*bpd)

    # Layer 0 is specific and has to be constructed separately
    for i in xrange(blocks_per_dim[0] * blocks_per_dim[0]):
        unit_parameters = create_basic_unit_v1(simulation_dict['learning_rates'][0], momentum, tau, additional_learning_rate)
        x = (i / blocks_per_dim[0])*input_block_size
        y = (i % blocks_per_dim[0])*input_block_size
        dx = input_block_size
        dy = input_block_size
        input_block = SharedArray.DynamicView(input_array_float)[x:x+dx, y:y+dy]
        predicted_block = SharedArray.DynamicView(simulation_dict['predicted_arrays'][0])[x:x+dx, y:y+dy]
        predicted_block_2 = SharedArray.DynamicView(simulation_dict['predicted_arrays_t2'][0])[x:x+dx, y:y+dy]
        delta_block = SharedArray.DynamicView(simulation_dict['delta_arrays'][0])[x:x+dx, y:y+dy]
        if not (predicted_block.shape == (dx, dy, 3)):
            print predicted_block.shape
            raise Exception("Block sizes don't agree")
        k = (i / blocks_per_dim[0])*hidden_size
        l = (i % blocks_per_dim[0])*hidden_size
        output_block = SharedArray.DynamicView(simulation_dict['state_arrays'][0])[k:k+hidden_size, l:l+hidden_size]
        unit_parameters['signal_blocks'].append((input_block,
                                                 delta_block,
                                                 predicted_block,
                                                 predicted_block_2
                                                 ))
        unit_parameters['output_block'] = output_block
        if readout_layer[0]:
            # Motor heatmap prediction
            layer = 0
            bpd = blocks_per_dim[layer]
            readout_teaching_block = SharedArray.DynamicView(simulation_dict['readout_arrays'][layer])[(i / bpd)*readout_block_size[0]:(i / bpd+1)*readout_block_size[0], (i % bpd)*readout_block_size[0]:(i % bpd+1)*readout_block_size[0]]
            readout_delta_block = SharedArray.SharedNumpyArray_like(readout_teaching_block)
            predicted_readout_block = SharedArray.DynamicView(simulation_dict['predicted_readout_arrays'][layer])[(i / bpd)*readout_block_size[0]:(i / bpd+1)*readout_block_size[0], (i % bpd)*readout_block_size[0]:(i % bpd+1)*readout_block_size[0]]
            unit_parameters['readout_blocks'] = [(readout_teaching_block, readout_delta_block, predicted_readout_block)]
            unit_parameters["layer"] = 0
            # End motor heatmap prediction
        simulation_dict['stage0'].append(unit_parameters)
    # Layer 0 surround
    gather_surround(simulation_dict, (base_index[0], blocks_per_dim[0]), radius=lateral_radius, context_factor=context_factor_lateral, exclude_self=exclude_self)

    # The following layers are more generic
    for layer in range(1, layers):
        for i in xrange(blocks_per_dim[layer] * blocks_per_dim[layer]):
            unit_parameters = create_basic_unit_v1(simulation_dict['learning_rates'][layer], momentum, tau, additional_learning_rate)
            k = (i / blocks_per_dim[layer])*hidden_size
            l = (i % blocks_per_dim[layer])*hidden_size
            output_block = SharedArray.DynamicView(simulation_dict['state_arrays'][layer])[k:k+hidden_size, l:l+hidden_size]
            unit_parameters['output_block'] = output_block
            if readout_layer[layer]:
                # Motor heatmap prediction
                bpd = blocks_per_dim[layer]
                readout_teaching_block = SharedArray.DynamicView(simulation_dict['readout_arrays'][layer])[(i / bpd)*readout_block_size[layer]:(i / bpd+1)*readout_block_size[layer], (i % bpd)*readout_block_size[layer]:(i % bpd+1)*readout_block_size[layer]]
                readout_delta_block = SharedArray.SharedNumpyArray_like(readout_teaching_block)
                predicted_readout_block = SharedArray.DynamicView(simulation_dict['predicted_readout_arrays'][layer])[(i / bpd)*readout_block_size[layer]:(i / bpd+1)*readout_block_size[layer], (i % bpd)*readout_block_size[layer]:(i % bpd+1)*readout_block_size[layer]]
                unit_parameters['readout_blocks'] = [(readout_teaching_block, readout_delta_block, predicted_readout_block)]
                unit_parameters["layer"] = layer
                # End motor heatmap prediction
            simulation_dict['stage0'].append(unit_parameters)
        # Connect to the previous layer
        connect_forward_and_back_v1(simulation_dict, (base_index[layer-1], blocks_per_dim[layer-1], simulation_dict['predicted_arrays'][layer], simulation_dict['predicted_arrays_t2'][layer]), (base_index[layer], blocks_per_dim[layer]), square_size=fan_in_square_size, radius=fan_in_radius, context_factor=context_factor_feedback)
        # Layer surround
        gather_surround(simulation_dict, (base_index[layer], blocks_per_dim[layer]), radius=lateral_radius, context_factor=context_factor_lateral, exclude_self=exclude_self)
        if send_context_two_layers_back and layer > 1:
            connect_back(simulation_dict, (base_index[layer], blocks_per_dim[layer]), (base_index[layer-2], blocks_per_dim[layer-2]), square_size=2*fan_in_square_size, radius=2*fan_in_radius, context_factor=context_factor_feedback)

    # Add the global feedback from the top layer
    if last_layer_context_to_all:
        logging.info("Connecting last layer back to everyone")
        for to_idx in xrange(base_index[layers-1]):
            for from_idx in range(base_index[layers-1], len(simulation_dict["stage0"])):
                context_block = simulation_dict['stage0'][from_idx]['output_block']
                delta_block2 = SharedArray.SharedNumpyArray_like(context_block)
                simulation_dict['stage0'][from_idx]['delta_blocks'].append(delta_block2)
                # Connect the context block to the source
                simulation_dict['stage0'][to_idx]['context_blocks'].append((context_block, delta_block2, context_factor_feedback))

    simulation_dict['stage0_size'] = len(simulation_dict['stage0'])
    for i in range(simulation_dict['stage0_size']):
            simulation_dict['stage0'][i]['flags'] = simulation_dict['flags']
            unit.ExecutionUnit.generate_missing_parameters(simulation_dict['stage0'][i], options=options)
    return simulation_dict


def upgrade_readout(simulation_dict):
    """
    Upgrade the dictionary with a parameters for a three layer perceptron for readout
    :param simulation_dict:
    :return:
    """
    logging.info("Upgrading readout network to a full perceptron")
    import PVM_framework.SharedArray as SharedArray
    import PVM_framework.MLP as MLP
    simulation_dict['stage0_size'] = len(simulation_dict['stage0'])
    needed = True
    for i in range(simulation_dict['stage0_size']):
        if len(simulation_dict['stage0'][i]["MLP_parameters_additional"]['layers']) == 2 and len(simulation_dict['stage0'][i]["MLP_parameters_additional"]['weights']) == 1:
            nhidden = simulation_dict['stage0'][i]["MLP_parameters_additional"]['layers'][0]['activation'].shape[0]-1
            nadditional = simulation_dict['stage0'][i]["MLP_parameters_additional"]['layers'][-1]['activation'].shape[0]-1
            layer = {'activation': SharedArray.SharedNumpyArray((nhidden+1), np.float),
                     'error': SharedArray.SharedNumpyArray((nhidden+1), np.float),
                     'delta': SharedArray.SharedNumpyArray((nhidden+1), np.float)
                     }

            simulation_dict['stage0'][i]["MLP_parameters_additional"]['layers'].insert(1, layer)
            simulation_dict['stage0'][i]["MLP_parameters_additional"]['weights'] = \
                [MLP.initialize_weights(SharedArray.SharedNumpyArray((nhidden+1, nhidden), np.float)),
                 MLP.initialize_weights(SharedArray.SharedNumpyArray((nhidden+1, nadditional), np.float)),
                 ]
        else:
            needed = False
    if needed:
        logging.info("Upgrade complete")
    else:
        logging.info("Upgrade was not nescessary")


def upgrade(simulation_dict):
    old_log = simulation_dict['error_log']
    if old_log.shape[1] < 1000000:
        new_log = SharedArray.SharedNumpyArray((old_log.shape[0], 1000000), np.float)
        new_log[:, 0:old_log.shape[1]] = old_log
        simulation_dict['error_log'] = new_log
    if "flags" not in simulation_dict.keys():
        simulation_dict['flags'] = SharedArray.SharedNumpyArray((16,), np.uint8)
        for i in range(16):
            simulation_dict['flags'][i] = 0
    if "record_filename" not in simulation_dict.keys():
        simulation_dict['record_filename'] = SharedArray.SharedNumpyArray((256,), np.uint8)


def apply_options(simulation_dict, options):
    if "bias_free" in options and options["bias_free"] == "1":
        for i in range(simulation_dict['stage0_size']):
            simulation_dict['stage0'][i]["Primary_Predictor_params"]['no_bias'] = True
            if "Residual_Predictor_params" in simulation_dict['stage0'][i].keys():
                simulation_dict['stage0'][i]["Residual_Predictor_params"]['no_bias'] = True
    if "bias_free_additional" in options and options["bias_free_additional"] == "1":
        for i in range(simulation_dict['stage0_size']):
            simulation_dict['stage0'][i]["Readout_Predictor_params"]['no_bias'] = True
    simulation_dict['options'] = options


def upgrade_dictionary_to_ver1_0(simulation_dict):
    upgrade(simulation_dict)
    if "version_major" not in simulation_dict.keys() or simulation_dict["version_major"] < 1:
        logging.info("Simulation dictionary of the old type, automatically upgrading to ver 1.0")
        if 'learning_rates' not in simulation_dict.keys():
            simulation_dict['learning_rates'] = []
        if 'momenta' not in simulation_dict.keys():
            simulation_dict['momenta'] = []
        if 'taus' not in simulation_dict.keys():
            simulation_dict['taus'] = []
        if 'predicted_arrays' not in simulation_dict.keys():
            simulation_dict['predicted_arrays'] = []
        if 'predicted_arrays_t2' not in simulation_dict.keys():
            simulation_dict['predicted_arrays_t2'] = []
        if 'predicted_readout_arrays' not in simulation_dict.keys():
            simulation_dict['predicted_readout_arrays'] = []
        if 'readout_arrays' not in simulation_dict.keys():
            simulation_dict['readout_arrays'] = []
        if 'predicted_arrays' not in simulation_dict.keys():
            simulation_dict['predicted_arrays'] = []
        if 'state_arrays' not in simulation_dict.keys():
            simulation_dict['state_arrays'] = []
        if 'delta_arrays' not in simulation_dict.keys():
            simulation_dict['delta_arrays'] = []
        for i in range(PVM_MAX_LAYERS):  # max number of layers
            if "delta_array%02d" % i in simulation_dict.keys():
                simulation_dict['delta_arrays'].append(simulation_dict['delta_array%02d' % i])
                del simulation_dict['delta_array%02d' % i]
            if "learning_rate%02d" % i in simulation_dict.keys():
                simulation_dict['learning_rates'].append(simulation_dict['learning_rate%02d' % i])
                del simulation_dict['learning_rate%02d' % i]
            if "state_array%02d" % i in simulation_dict.keys():
                simulation_dict['state_arrays'].append(simulation_dict['state_array%02d' % i])
                del simulation_dict['state_array%02d' % i]
            if "predicted_readout_array_float%02d" % i in simulation_dict.keys():
                simulation_dict['predicted_readout_arrays'].append(simulation_dict['predicted_readout_array_float%02d' % i])
                del simulation_dict['predicted_readout_array_float%02d' % i]
            if "readout_array_float%02d" % i in simulation_dict.keys():
                simulation_dict['readout_arrays'].append(simulation_dict['readout_array_float%02d' % i])
                del simulation_dict['readout_array_float%02d' % i]
            if "predicted_array%02d" % i in simulation_dict.keys():
                simulation_dict['predicted_arrays'].append(simulation_dict['predicted_array%02d' % i])
                simulation_dict['predicted_arrays_t2'].append(SharedArray.SharedNumpyArray_like(simulation_dict['predicted_array%02d' % i]))
                del simulation_dict['predicted_array%02d' % i]

        if "motor_delta_array_float" in simulation_dict.keys():
            del simulation_dict["motor_delta_array_float"]
        if "readout_array_float" in simulation_dict.keys():
            del simulation_dict["readout_array_float"]
        if "readout_array_float" in simulation_dict.keys():
            del simulation_dict["readout_array_float"]
        if "predicted_motor_derivative_array_float" in simulation_dict.keys():
            del simulation_dict["predicted_motor_derivative_array_float"]
        if "predicted_readout_array_float" in simulation_dict.keys():
            del simulation_dict["predicted_readout_array_float"]
        if "additional_learning_rate" in simulation_dict.keys():
            simulation_dict["readout_learning_rate"] = simulation_dict["additional_learning_rate"]
            del simulation_dict["additional_learning_rate"]
        elif "readout_learning_rate" not in simulation_dict.keys():
            simulation_dict["readout_learning_rate"] = SharedArray.SharedNumpyArray((1, ), np.float)
            simulation_dict["readout_learning_rate"][:] = 0.00001

        simulation_dict['execution_unit_module'] += "_v1"
        if not simulation_dict['execution_unit_module'].startswith("PVM_models"):
            simulation_dict['execution_unit_module'] = "PVM_models."+simulation_dict['execution_unit_module']
        ex_unit = importlib.import_module(simulation_dict['execution_unit_module'])
        for s in range(simulation_dict['stages']):
            stage = simulation_dict['stage%d' % s]
            for unit in stage:
                signal_blocks = unit['signal_blocks']
                unit['signal_blocks'] = []
                for block in signal_blocks:
                    # Each block: signal_block, delta_block, prediction_t+1, prediction_t+2
                    unit['signal_blocks'].append([block[0], block[1], block[2], SharedArray.SharedNumpyArray_like(block[2])])
                context_blocks = unit['context_blocks']
                unit['context_blocks'] = []
                for block in context_blocks:
                    # Each block: context_block, delta_block, switching_factor
                    unit['context_blocks'].append([block[0], block[1], block[2]])
                readout_blocks = unit['predicted_blocks']
                del unit['predicted_blocks']
                unit['readout_blocks'] = []
                for block in readout_blocks:
                    # Each block: teaching_signal, delta_block, readout_block
                    unit['readout_blocks'].append([block[0], block[1], block[2]])
                # Delta blocks can remain unchanged
                if "learning_rate" in unit.keys():
                    unit['primary_learning_rate'] = unit.pop("learning_rate")
                if "momentum" in unit.keys():
                    unit['primary_momentum'] = unit.pop("momentum")
                    unit['readout_momentum'] = unit['primary_momentum']
                if "additional_learning_rate" in unit.keys():
                    unit['readout_learning_rate'] = unit.pop("additional_learning_rate")
                else:
                    unit['readout_learning_rate'] = simulation_dict["readout_learning_rate"]
                # Output block may remain unchanged
                if "MLP_parameters" in unit.keys():
                    unit['Primary_Predictor_params'] = unit.pop("MLP_parameters")
                if "MLP_parameters_res" in unit.keys():
                    unit['Residual_Predictor_params'] = unit.pop("MLP_parameters_res")
                if "MLP_parameters_additional" in unit.keys():
                    unit['Readout_Predictor_params'] = unit.pop("MLP_parameters_additional")
                unit['flags'] = simulation_dict['flags']
                ex_unit.ExecutionUnit.upgrade_to_ver_1(unit)

        simulation_dict['version_major'] = 1
        simulation_dict['version_minor'] = 0
        # Remove all the old source files
        simulation_dict['sources'] = {}
        logging.info("Upgrade succesfull")
    else:
        for s in range(simulation_dict['stages']):
            stage = simulation_dict['stage%d' % s]
            for unit in stage:
                unit['flags'] = simulation_dict['flags']
        logging.info("Dictionary already ver 1.0 or above, no need to upgrade")


if __name__ == "__main__":
    pass
