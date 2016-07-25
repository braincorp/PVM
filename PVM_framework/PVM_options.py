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
import textwrap

option_default = {
    "initial_learning_rate": "0.0003",
    "final_learning_rate": "0.0001",
    "delay_each_layer_learning": "100000",
    "delay_final_learning_rate": "100000",
    "enable_lateral_at": "900000",
    "enable_feedback_at": "900000",
    "input_block_size": "5",
    "hidden_block_size": "5",
    "layer_shapes": ["4", "3", "2", "1"],
    "readout_block_size": ["1", "1", "2", "4"],
    "enable_readout": ["1", "1", "1", "1"],
    "lateral_radius": "1.5",
    "context_exclude_self": "0",
    "fan_in_square_size": "2",
    "fan_in_radius": "2",
    "readout_depth": "1",
    "reverse": "0",
    "new_name": "",
    "polynomial": "0",
    "autoencoder": "0",
    "model_type": "small",
    "ex_module": "PVM_models.PVM_unit_v1",
    "unit_type": "simple",
    "stereo": "0",
    "only_one_channel": "0",
    "supervised": "0",
    "supervised_rate": "0.0002",
    "steps": "100000000",
    "bias_free": "0",
    "bias_free_readout": "0",
    "disable_lateral": "0",
    "disable_feedback": "0",
    "dataset": "short",
    "last_layer_context_to_all": "1",
    "send_context_two_layers_back": "0",
    "version_major": "0",
    "version_minor": "0",
    "use_t_minus_2_block": "0",
    "use_derivative": "1",
    "use_integral": "1",
    "use_error": "1",
    "predict_two_steps": "0",
    "use_global_backprop": "0",
    "feed_context_in_complex_layer": "0",
    "tau": "0.98",
    "momentum": "0.5",
    "normalize_output": "0",
    "save_source_files": "0",
    "backpropagate_readout_error": "0",
}

option_descriptions = {
    "initial_learning_rate": "Learning rate used in the initial stage of the simulation.",
    "final_learning_rate": "Learning rate used in the remaining part of the simulation.",
    "delay_each_layer_learning": "Enable learning in consequtive layers by this number of steps.",
    "delay_final_learning_rate": "Switch to final learning rate by this number of steps.",
    "enable_lateral_at": "Enable lateral connections at this step.",
    "enable_feedback_at": "Enable feedback at this step.",
    "input_block_size": "Size of the input tile.",
    "hidden_block_size": "Size of the internal representation.",
    "layer_shapes": "List of layer sizes from the input layer towards higher layers.",
    "readout_block_size": "Dimensions of he readout blocks in each layer.",
    "enable_readout": "Determines in which layers the readout is performed.",
    "lateral_radius": "Radius of the lateral interactions, 1 - 4 neighbourhood, 1.5 - 8 neighbourhood, 2 - 12 neighbourhood. See PVM_Create.py for details.",
    "context_exclude_self": "Exclude self as a source of context.",
    "fan_in_square_size": "Size of the square from which the fan ins will be selected. See PVM_Create for details.",
    "fan_in_radius": "Radius in within which the fan ins will be selected.",
    "readout_depth": "Depth of the readout classifier.",
    "reverse": "Run the input dataset in reverse order.",
    "new_name": "Set the new name for a simulation, subsequet snapshots will be saved in a new S3 directory.",
    "polynomial": "Use the polynomial sigmoid function in the MLP (slightly faster computation).",
    "autoencoder": "For executions modules that allow, use autoencoder instead of predictive encoder, supported by PVM_unit.",
    "model_type": "The architecture name, choose from tiny, small2, small and large.",
    "ex_module": "Name of the python file (without .py) that will be used as the execution unit e.g. PVM_unit.",
    "unit_type": "For execution modules that support choose from simple or complex. Simple is typically a 3 layer MLP, complex is typically 4 layer MLP.",
    "stereo": "Use an interlaced stereo video for input.",
    "only_one_channel": "Use only one channel from a stereo dataset (interlaced with black frame).",
    "supervised": "Run the simulation in supervised mode.",
    "supervised_rate": "Learning rate for the readout classifier.",
    "steps": "Number of steps to run for.",
    "bias_free": "Use a biasfree perceptron (with bias unit equal zero).",
    "bias_free_readout": "Use bias free perceptron for the readout classifier.",
    "disable_lateral": "Do not enable lateral context.",
    "disable_feedback": "Do not enable feedback context.",
    "dataset": "Name of the dataset to use.",
    "last_layer_context_to_all": "Send last layer activations as context to every layer below.",
    "send_context_two_layers_back": "Send context two layers backwards in addition to normal context one layer backwards.",
    "version_major": "Major version number of the dictionary.",
    "version_minor": "Minor version number of the dictionary.",
    "use_t_minus_2_block": "Use t-2 in addition to t-1 as predictor input, supported by PVM_unit.",
    "use_derivative": "Use difference between t-1 and t-2 as additional features, supported by PVM_unit.",
    "use_integral": "Use integral trace as additional feature with value of time constant given in tau parameter, supported by PVM_unit.",
    "use_error": "Use error of the previous prediction as additional feature, supported by PVM_unit.",
    "predict_two_steps": "Predict two steps in the future, supported by PVM_unit.",
    "use_global_backprop": "Backpropagate error through context and feedforward connections outside of the unit, supported by PVM_unit.",
    "feed_context_in_complex_layer": "Put context activation in the second layer of the complex unit.",
    "tau": "Time constant used to compute the integral trace feature.",
    "momentum": "Default momentum value used in the backprop algorithm.",
    "normalize_output": "Use simple running min-max normalization on the output of the predictive encoder compressed features.",
    "save_source_files": "Save all *.py and *pyx files from the folder where the simulation was created for future inspection.",
    "backpropagate_readout_error": "Propagate the error of the readout classifier into the predictive encoder"
}

option_values = {
    "initial_learning_rate": "float number",
    "final_learning_rate": "float number",
    "delay_each_layer_learning": "int number",
    "delay_final_learning_rate": "int number",
    "enable_lateral_at": "int number",
    "enable_feedback_at": "int number",
    "input_block_size": "int number",
    "hidden_block_size": "int number",
    "layer_shapes": "list of int numbers",
    "readout_block_size": "list of int numbers",
    "enable_readout": "list of int numbers",
    "lateral_radius": "float number",
    "context_exclude_self": ("0", "1"),
    "fan_in_square_size": "int number",
    "fan_in_radius": "int number",
    "readout_depth": "int number",
    "reverse": ("0", "1"),
    "new_name": "str",
    "polynomial": ("0", "1"),
    "autoencoder": ("0", "1"),
    "model_type": ("tiny", "small2", "small", "large"),
    "ex_module": ("PVM_unit", "PVM_unit_2step", "PVM_unit_2step_residual"),
    "unit_type": ("simple", "complex"),
    "stereo": ("0", "1"),
    "only_one_channel": ("0", "1"),
    "supervised": ("0", "1"),
    "supervised_rate": "float number",
    "steps": "int number",
    "bias_free": ("0", "1"),
    "bias_free_readout": ("0", "1"),
    "disable_lateral": ("0", "1"),
    "disable_feedback": ("0", "1"),
    "dataset": "str",
    "last_layer_context_to_all": ("0", "1"),
    "send_context_two_layers_back": ("0", "1"),
    "version_major": "int number",
    "version_minor": "int number",
    "use_t_minus_2_block": ("0", "1"),
    "use_derivative": ("0", "1"),
    "use_integral": ("0", "1"),
    "use_error": ("0", "1"),
    "predict_two_steps": ("0", "1"),
    "use_global_backprop": ("0", "1"),
    "feed_context_in_complex_layer": ("0", "1"),
    "tau": "float number",
    "momentum": "float number",
    "normalize_output": ("0", "1"),
    "save_source_files": ("0", "1"),
    "backpropagate_readout_error": ("0", "1"),
}


def get_option_help():
    wr = textwrap.TextWrapper(initial_indent='  ', subsequent_indent='                       ', width=70)
    help_str = "Available options (given as json dict either from the command line -O or in a file via -S):\n"
    help_str += "\n"
    for option in option_default.keys():
        option_str = ""
        option_str += " %s - default value %s. " % (option, option_default[option])
        option_str += " %s Possible values: " % (option_descriptions[option])
        if type(option_values[option]) == tuple:
            for vals in option_values[option]:
                option_str += "\"%s\"  " % vals
        else:
            option_str += option_values[option] + "\n"
        help_str += wr.fill(option_str) + "\n"
    return help_str


def parse_options(options_given, options_in_the_dict=None):
    parsed = option_default.copy()
    for option in option_default.keys():
        if options_in_the_dict is not None and option in options_in_the_dict:
            parsed[option] = options_in_the_dict[option]
        if option in options_given:
            parsed[option] = options_given[option]
    for option in options_given.keys():
        if option not in option_default.keys():
            if option == "bias_free_additional":
                parsed['bias_free_readout'] = options_given["bias_free_additional"]
            else:
                raise Exception("Unknown option %s given" % option)
    return parsed
