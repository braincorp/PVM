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
import time
import cv2
import numpy as np
import PVM_framework.CoreUtils as CoreUtils
import PVM_framework.PVM_Create as PVM_Create
import PVM_framework.AbstractExecutionManager as AbstractExecutionManager
import os
import logging
import PVM_framework.PVM_display_helper as DisplayHelper
import sys


def get_square_res(prop_dict):
    """
    Small helper function to derive the maximal dimension of state/input arrays in
    a PVM dictionary.
    :param prop_dict: PVM simulation dict
    :return: int
    """
    max_d = 0
    for i in range(PVM_Create.PVM_MAX_LAYERS):
        if "state_array%02d" % i in prop_dict.keys():
            max_d = max(max_d, np.max(prop_dict["state_array%02d" % i].shape))
    max_d = max(max_d, np.max(prop_dict["input_array"].shape))
    max_d = max(max_d, 128)
    return max_d


class Manager(AbstractExecutionManager.ExecutionManager):
    """
    Manager class that does all the housekeeping while the PVM model is running.
    Housekeeping includes supplying input, performing display operations,
    triggering modes requested by the debug console etc.
    """
    def __init__(self,
                 prop_dict,
                 steps_to_run,
                 signal_provider,
                 record=False,
                 video_recorder=None,
                 do_display=False,
                 checkpoint=False,
                 checkpoint_storage=None,
                 evaluate=False,
                 frames=-1,
                 collect_error=False,
                 dataset_name=""):

        self.prop_dict = prop_dict
        self.signal = signal_provider
        self.video_recorder = video_recorder
        self.steps_to_run = steps_to_run
        self._running = True
        self.num_layers = len(prop_dict["state_arrays"])
        logging.info("Loading a model with %d layers" % self.num_layers)
        self.display = DisplayHelper.DisplayHelperObject(width=(get_square_res(self.prop_dict)+16) * (2 + self.num_layers)+16,
                                                         height=(get_square_res(self.prop_dict)+16) * 5+16,
                                                         grid=(2 + self.num_layers, 5))
        self.record = record
        self.do_display = do_display
        self.update_interval = 1
        self.steps = 0
        self.frame_to_drop = frames
        self.t_start = time.time()
        self.t_prev = time.time()
        self.readout_targets = [None] * (self.num_layers)
        self.evaluate = evaluate
        self.checkpoint = checkpoint
        self.checkpoint_storage = checkpoint_storage
        self.error_accumulator = np.zeros((self.num_layers,))
        self.error_accumulator_reconstruction = np.zeros((self.num_layers,))
        self.collect_error = collect_error
        self.predictive_error = 0
        self.predictive_error_array = []
        self.reconstruction_error = 0
        self.reconstruction_error_array = []
        self.actual_input_prev = np.zeros_like(self.prop_dict['predicted_arrays'][0].view(np.ndarray))
        self.actual_input_prev_2 = np.zeros_like(self.prop_dict['predicted_arrays'][0].view(np.ndarray))
        self.state_prev = []
        self.dataset_name = dataset_name
        for layer_num in range(1, self.num_layers):
            key_state_below = "state_arrays"
            self.state_prev.append(np.zeros_like(self.prop_dict[key_state_below][(layer_num-1)].view(np.ndarray)))
        self.prev_now = time.time()
        self.actual_input = np.zeros_like(self.prop_dict['predicted_arrays'][0].view(np.ndarray))
        self.dream_experiment = False
        self.dream_experiment_data = {}
        self.learning_rate_buffer = None
        self.readout_learning_rate_buffer = None
        self.frozen_learning = False
        self.freeze_learning_scheduled = False
        self.unfreeze_learning_scheduled = False

    def _get_readout_targets(self, channel_name="mask"):
        """
        Get the readout training (supervisory) signals from the signal stream, resizing them
        as necessary for the size of the readout array.
        """
        result = []
        past_index = 0
        for layer_number in range(self.num_layers):
            past_index -= 0  # change to not go back to past but make the readout fully predictive.
            key = "readout_arrays"
            mask = cv2.resize(self.signal.get_signal(name=channel_name, time=past_index), dsize=self.prop_dict[key][layer_number].shape[:2][::-1])
            result.append((mask.astype(np.float)/255)*0.6+0.4)
        return result

    def start(self):
        """
        This will be called right before the simulation starts
        """
        self.signal.start()
        self.t_start = time.time()
        # Get the initial data
        self.current_frame = self.signal.get_signal(name="frame", time=0)
        self.readout_targets = self._get_readout_targets(channel_name="mask")

        if self.steps_to_run < 0:
            self.steps_to_run = self.signal.get_length()
        if "error_log" not in self.prop_dict.keys():
            error_log = np.zeros((self.num_layers+1, 20000), np.float)
            self.prop_dict['error_log'] = error_log

    def fast_action(self):
        """
        This is the time between steps of execution
        Data is consistent, but keep this piece absolutely minimal
        """
        # switch buffers
        if not self.evaluate:
            for i in range(self.num_layers):
                key = "readout_arrays"
                self.prop_dict[key][i][:] = self.readout_targets[i].astype(np.float).reshape(self.prop_dict[key][i].shape)

        self.actual_input_prev_2[:] = self.actual_input_prev
        self.actual_input_prev[:] = self.actual_input
        self.prop_dict['input_array'][:] = self.current_frame
        self.prop_dict['input_array_float'][:] = self.current_frame.astype(np.float)/255
        self.predicted = self.prop_dict['predicted_arrays'][0].view(np.ndarray).copy()
        self.actual_input = self.prop_dict['input_array_float'].view(np.ndarray).copy()
        self.calculate_errors()
        if self.checkpoint and int(self.prop_dict['N'][0]) % 100000 == 0:
            self.take_snapshot_and_backup()
        if self.freeze_learning_scheduled:
            self.freeze_learning_synchronised()
        if self.unfreeze_learning_scheduled:
            self.un_freeze_learning_synchronised()

    def calculate_errors(self):
        # compute the reconstruction error
        if self.collect_error:
            self.adifr = []
            self.adifr.append(np.abs(self.predicted - self.actual_input_prev_2))
            for layer_num in range(1, self.num_layers):
                self.adifr.append(np.abs(self.prop_dict["predicted_arrays"][layer_num].view(np.ndarray)-self.state_prev[layer_num-1]))

        # This is for predictive error
        self.adif = []
        self.adif.append(np.abs(self.predicted - self.actual_input_prev))
        for layer_num in range(1, self.num_layers):
            self.state_prev[layer_num-1][:] = self.prop_dict["state_arrays"][layer_num-1].view(np.ndarray)
            self.adif.append(np.abs(self.prop_dict["predicted_arrays"][layer_num].view(np.ndarray) - self.prop_dict["state_arrays"][layer_num-1]))

        for layer_num in range(self.num_layers):
            self.error_accumulator[layer_num] += np.sum(self.adif[layer_num], dtype=np.float)/np.prod(self.adif[layer_num].shape)
            if self.collect_error:
                self.error_accumulator_reconstruction[layer_num] += np.sum(self.adifr[layer_num], dtype=np.float)/np.prod(self.adifr[layer_num].shape)
        if self.collect_error:
            self.predictive_error += np.sum(np.array(self.error_accumulator))
            self.predictive_error_array.append(np.sum(np.array(self.error_accumulator)))
            self.reconstruction_error += np.sum(np.array(self.error_accumulator_reconstruction))
            self.reconstruction_error_array.append(np.sum(np.array(self.error_accumulator_reconstruction)))

        if int(self.prop_dict['N'][0]) % PVM_Create.PVM_LOG_ERROR_EVERY == 0:
            log_step = (self.prop_dict['N'][0]/PVM_Create.PVM_LOG_ERROR_EVERY)-1
            self.prop_dict['error_log'][0, log_step] = self.prop_dict['N'][0]
            for l in range(1, 1+self.num_layers, 1):
                self.prop_dict['error_log'][l, log_step] = self.error_accumulator[l-1]/(255*PVM_Create.PVM_LOG_ERROR_EVERY)
            self.error_accumulator *= 0
            self.error_accumulator_reconstruction *= 0

    def take_snapshot_and_backup(self):
        if self.prop_dict['readout_learning_rate'][0] == 0:
            # Unsupervised
            CoreUtils.save_model(self.prop_dict, "PVM_failsafe_%010d.p.gz" % int(self.prop_dict['N'][0]))
            to_folder = "PVM_models/%s_%s_%s" % (self.prop_dict['timestamp'], self.prop_dict['name'], self.prop_dict['hash'])
            from_path = "./PVM_failsafe_%010d.p.gz" % int(self.prop_dict['N'][0])
            logging.info("Uploading %s/%s" % (to_folder, from_path[2:]))
            self.checkpoint_storage.put(from_path=from_path, to_folder=to_folder, overwrite=True)
            os.remove("./PVM_failsafe_%010d.p.gz" % int(self.prop_dict['N'][0]))
        else:
            # Supervised
            self.signal.reset()  # To avoid dataset aliasing
            CoreUtils.save_model(self.prop_dict, "PVM_state_supervised_%s_%d_%d_%f.p.gz" % (self.dataset_name, self.prop_dict['N'][0], int(self.steps), float(self.prop_dict['readout_learning_rate'][0])))
            to_folder = "PVM_models/%s_%s_%s" % (self.prop_dict['timestamp'], self.prop_dict['name'], self.prop_dict['hash'])
            from_path = "./PVM_state_supervised_%s_%d_%d_%f.p.gz" % (self.dataset_name, self.prop_dict['N'][0], int(self.steps), float(self.prop_dict['readout_learning_rate'][0]))
            logging.info("Uploading %s/%s" % (to_folder, from_path[2:]))
            self.checkpoint_storage.put(from_path=from_path, to_folder=to_folder, overwrite=True)
            os.remove(from_path)

    def construct_display(self):
            """
            Creates the display window.
            :return:
            """
            motor_size = self.prop_dict['predicted_readout_arrays'][0].shape[:2][::-1]
            max_readout_array = np.maximum.reduce([cv2.resize(self.prop_dict['predicted_readout_arrays'][i].view(np.ndarray), dsize=motor_size) for i in range(self.num_layers)])
            am = np.unravel_index(np.argmax(max_readout_array), max_readout_array.shape)
            if len(am) == 3:
                for d in range(3):
                    if am[2] != d:
                        max_readout_array[:, :, d] = 0

            heatmap = max_readout_array.view(np.ndarray)
            heatmap = cv2.resize(heatmap, dsize=(self.prop_dict['input_array'].shape[1], self.prop_dict['input_array'].shape[0]))
            heatmap = np.maximum((heatmap - 0.4), 0)
            heatmap /= 0.6
            image = cv2.resize(self.signal.get_signal(name="frame", time=-3), dsize=(self.prop_dict['input_array'].shape[1], self.prop_dict['input_array'].shape[0]))
            image2 = image.copy()
            if len(heatmap.shape) == 3:
                heatmap = np.max(heatmap, axis=2)
            (x, y) = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            # Bull's eye
            if heatmap[x, y] > 0.4:
                cv2.circle(image2, (y, x), 12, color=(255, 255, 255), thickness=1, lineType=cv2.CV_AA)
                cv2.circle(image2, (y, x), 10, color=(0, 0, 0), thickness=1, lineType=cv2.CV_AA)
                cv2.circle(image2, (y, x), 22, color=(255, 255, 255), thickness=1, lineType=cv2.CV_AA)
                cv2.circle(image2, (y, x), 20, color=(0, 0, 0), thickness=1, lineType=cv2.CV_AA)
            if heatmap[x, y] > 0.6:
                cv2.circle(image2, (y, x), 12, color=(255, 255, 255), thickness=2, lineType=cv2.CV_AA)
                cv2.circle(image2, (y, x), 10, color=(0, 0, 0), thickness=2, lineType=cv2.CV_AA)
                cv2.circle(image2, (y, x), 22, color=(255, 255, 255), thickness=2, lineType=cv2.CV_AA)
                cv2.circle(image2, (y, x), 20, color=(0, 0, 0), thickness=2, lineType=cv2.CV_AA)

            image = image.astype(float)/255
            image[:, :, 0] *= heatmap
            image[:, :, 1] *= heatmap
            image[:, :, 2] *= heatmap

            # Construct the window
            self.display.place_image(0, 0, self.actual_input_prev, "Input frame")
            self.display.place_image(0, 1, self.predicted, "Predicted image")
            self.display.place_image(0, 2, self.adif[0], "State-pred.")
            if self.collect_error:
                self.display.place_image(0, 4, self.adifr[0], "Rec Difference")
            for layer_num in range(0, self.num_layers-1):
                self.display.place_image(1 + layer_num, 0, self.prop_dict['state_arrays'][layer_num].view(np.ndarray), "%dL state" % (layer_num+1))
                self.display.place_image(1 + layer_num, 1, self.prop_dict['predicted_arrays'][layer_num+1].view(np.ndarray), "%dL pred. state" % (layer_num+1))
                self.display.place_image(1 + layer_num, 2, self.adif[layer_num+1], "State - pred.")
                self.display.place_image(0 + layer_num, 4, self.prop_dict['predicted_arrays_t2'][layer_num].view(np.ndarray), "%dL pred state t+1" % (layer_num))
                if self.collect_error:
                    self.display.place_image(1 + layer_num, 4, self.adifr[layer_num+1], "Rec Difference")
                self.display.place_image(1 + layer_num, 3, self.prop_dict['predicted_readout_arrays'][layer_num].view(np.ndarray), "Readout %d" % layer_num)
            self.display.place_image(self.num_layers-1, 4, self.prop_dict['predicted_arrays_t2'][self.num_layers-1].view(np.ndarray), "%dL pred state t+1" % (self.num_layers-1))
            self.display.place_image(self.num_layers, 0, self.prop_dict['state_arrays'][self.num_layers-1].view(np.ndarray), "%dL state" % self.num_layers)
            self.display.place_image(self.num_layers, 3, self.prop_dict['predicted_readout_arrays'][self.num_layers-1].view(np.ndarray), "Readout %d" %(self.num_layers-1))
            self.display.place_image(self.num_layers, 1, max_readout_array.view(np.ndarray), "Combined readout")
            self.display.clear_cell(self.num_layers+1, 3)
            self.display.place_color_logo(self.num_layers+1, 3)
            self.display.place_text(self.num_layers+1, 3, voffset=50, text=self.prop_dict['name'])
            self.display.place_text(self.num_layers+1, 3, voffset=60, text=self.prop_dict['hash'])
            self.display.place_text(self.num_layers+1, 3, voffset=70, text=self.prop_dict['timestamp'])

            now = time.time()
            interval = now - self.t_prev
            self.t_prev = now
            self.display.place_text(self.num_layers+1, 3, voffset=80, text="N=%d fps=%2.2f" % (self.prop_dict['N'][0], 1 / interval))
            self.display.place_text(self.num_layers+1, 3, voffset=90, text=time.ctime())

            self.display.place_text(self.num_layers+1, 3, voffset=100, text="Learning rate[0] %2.5f" % self.prop_dict['learning_rates'][0][0])
            self.display.place_text(self.num_layers+1, 3, voffset=110, text="Lat.conn %s, fb.conn %s " % ("yes" if self.prop_dict['context_factor_lateral'][0] > 0 else "no", "yes" if self.prop_dict['context_factor_feedback'][0] > 0 else "no"))
            self.display.place_text(self.num_layers+1, 3, voffset=120, text="Dream mode %s" % ("yes" if self.prop_dict['flags'][0] == 1 else "no"))
            self.display.place_image(self.num_layers+1, 0, image, "Superimposed")
            self.display.place_image(self.num_layers+1, 2, heatmap, "Heatmap")
            self.display.place_image(self.num_layers+1, 1, image2.astype(np.uint8), "Tracking")

    def developmental_check(self):
        # Developmental stuff
        for i in range(self.num_layers):
            if self.prop_dict["N"][0] == int(self.prop_dict["options"]["delay_each_layer_learning"])*i:
                self.prop_dict['learning_rates'][i][0] = float(self.prop_dict["options"]["initial_learning_rate"])
            if self.prop_dict["N"][0] == int(self.prop_dict["options"]["delay_final_learning_rate"]) + \
                    int(self.prop_dict["options"]["delay_each_layer_learning"])*i:
                self.prop_dict['learning_rates'][i][0] = float(self.prop_dict["options"]["final_learning_rate"])

        if "disable_lateral" not in self.prop_dict.keys() and self.prop_dict["N"][0] == int(self.prop_dict["options"]["enable_lateral_at"]):
                self.prop_dict['context_factor_lateral'][0] = 1.0
        if "disable_feedback" not in self.prop_dict.keys() and self.prop_dict["N"][0] == int(self.prop_dict["options"]["enable_feedback_at"]):
            self.prop_dict['context_factor_feedback'][0] = 1.0

    def slow_action(self):
        """
        This is while the workers are running. You may do a lot of work here
        (preferably not more than the time of execution of workers).
        """
        self.developmental_check()
        self.readout_targets = self._get_readout_targets(channel_name="mask")
        self.current_frame = self.signal.get_signal(name="frame", time=0).copy()
        self.process_flags()
        if (self.do_display and self.steps % self.update_interval == 0) or self.record or self.frame_to_drop > 0:
            self.construct_display()
            # Display the window
            if self.do_display:
                cv2.imshow("PVM display", self.display.frame)
            if self.record:
                self.video_recorder.record(self.display.frame)
            if self.frame_to_drop>0 and self.steps==self.frame_to_drop:
                cv2.imwrite(self.rec_filename, self.display.frame)
            if self.do_display:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0
            if key == 27 or self.steps > self.steps_to_run:
                self._running = False
            if key == ord("a"):
                self.update_interval = max(1, self.update_interval-1)
            if key == ord("s"):
                self.update_interval += 1
        if self.dream_experiment:
            self.process_dream_experiment()
        now = time.time()
        sps = self.steps / (now-self.t_start)
        if now - self.prev_now > 2:
            print ("%3.3f steps per sec" % (sps)) + "\r",
            sys.stdout.flush()
            self.prev_now = now
        self.steps += 1
        if self.steps > self.steps_to_run:
            self._running = False
        self.signal.advance()

    def process_flags(self):
        if 'flags' in self.prop_dict.keys():
            if self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_DREAM:  # Dream mode
                self.current_frame[:] = (self.predicted*255).astype(np.uint8)
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_BLINDSPOT:  # Blind spot
                self.current_frame = self.signal.get_signal(name="frame", time=0).copy()
                indx0 = self.current_frame.shape[0]/2-self.current_frame.shape[0]/4
                indx1 = indx0 + self.current_frame.shape[0]/2
                indy0 = self.current_frame.shape[1]/2-self.current_frame.shape[0]/4
                indy1 = indy0 + self.current_frame.shape[0]/2
                self.current_frame[indx0:indx1, indy0:indy1] = 127
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_GRAY:  # Neutral input
                self.current_frame[:] = 127
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_NOISE:  # White noise
                self.current_frame[:] = np.random.randint(0, 255, self.current_frame.shape)
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_NOISE_SPOT:  # Noise spot
                self.current_frame = self.signal.get_signal(name="frame", time=0).copy()
                indx0 = self.current_frame.shape[0]/2-self.current_frame.shape[0]/4
                indx1 = indx0 + self.current_frame.shape[0]/2
                indy0 = self.current_frame.shape[1]/2-self.current_frame.shape[0]/4
                indy1 = indy0 + self.current_frame.shape[0]/2
                self.current_frame[indx0:indx1, indy0:indy1] = np.random.randint(0, 255, (indx1-indx0, indy1-indy0, 3))
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_INV_BLINDSPOT:  # Inverse blindspot
                self.current_frame[:] = 127
                indx0 = self.current_frame.shape[0]/2-self.current_frame.shape[0]/4
                indx1 = indx0 + self.current_frame.shape[0]/2
                indy0 = self.current_frame.shape[1]/2-self.current_frame.shape[0]/4
                indy1 = indy0 + self.current_frame.shape[0]/2
                self.current_frame[indx0:indx1, indy0:indy1] = self.signal.get_signal(name="frame", time=0).copy()[indx0:indx1, indy0:indy1]
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_INV_DREAM_SPOT:  # Partial dream
                self.current_frame[:] = (self.predicted*255).astype(np.uint8)
                indx0 = self.current_frame.shape[0]/2-self.current_frame.shape[0]/4
                indx1 = indx0 + self.current_frame.shape[0]/2
                indy0 = self.current_frame.shape[1]/2-self.current_frame.shape[0]/4
                indy1 = indy0 + self.current_frame.shape[0]/2
                self.current_frame[indx0:indx1, indy0:indy1] = self.signal.get_signal(name="frame", time=0).copy()[indx0:indx1, indy0:indy1]
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_BLINKS:  # Blinks
                if self.prop_dict["N"][0] % 20 == 0 or self.prop_dict["N"][0] % 20 == 1:
                    self.current_frame[:] = (self.predicted*255).astype(np.uint8)
                else:
                    self.current_frame[:] = self.signal.get_signal(name="frame", time=0).copy()
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_NOISY_SIGNAL:  # Noisy signal
                self.current_frame[:] = self.signal.get_signal(name="frame", time=0).copy() - 32+np.random.randint(0, 64, self.current_frame.shape)
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_INV_NOISE_SPOT:  # Inverse spot noise
                self.current_frame[:] = np.random.randint(0, 255, self.current_frame.shape)
                indx0 = self.current_frame.shape[0]/2-self.current_frame.shape[0]/4
                indx1 = indx0 + self.current_frame.shape[0]/2
                indy0 = self.current_frame.shape[1]/2-self.current_frame.shape[0]/4
                indy1 = indy0 + self.current_frame.shape[0]/2
                self.current_frame[indx0:indx1, indy0:indy1] = self.signal.get_signal(name="frame", time=0).copy()[indx0:indx1, indy0:indy1]
            elif self.prop_dict['flags'][0] == PVM_Create.PVM_FLAG_VAL_DREAM_SPOT:  # Dream spot
                self.current_frame = self.signal.get_signal(name="frame", time=0).copy()
                indx0 = self.current_frame.shape[0]/2-self.current_frame.shape[0]/4
                indx1 = indx0 + self.current_frame.shape[0]/2
                indy0 = self.current_frame.shape[1]/2-self.current_frame.shape[0]/4
                indy1 = indy0 + self.current_frame.shape[0]/2
                self.current_frame[indx0:indx1, indy0:indy1] = (self.predicted*255).astype(np.uint8)[indx0:indx1, indy0:indy1]

            if self.prop_dict['flags'][PVM_Create.PVM_LEARNING_FLAG] == PVM_Create.PVM_LEARNING_FREEZE:
                self.freeze_learning()
                self.prop_dict['flags'][PVM_Create.PVM_LEARNING_FLAG] = PVM_Create.PVM_LEARNING_RESET
            if self.prop_dict['flags'][PVM_Create.PVM_LEARNING_FLAG] == PVM_Create.PVM_LEARNING_UNFREEZE:
                self.un_freeze_learning()
                self.prop_dict['flags'][PVM_Create.PVM_LEARNING_FLAG] = PVM_Create.PVM_LEARNING_RESET
            if self.prop_dict['flags'][1] == 1:
                self.prop_dict['flags'][1] = 0
                self.dream_experiment = not self.dream_experiment
                logging.info("Dream experiment has been sheduled")
        if "finished" in self.prop_dict.keys():
            if self.prop_dict['finished'][0] == PVM_Create.PVM_FLAG_VAL_TRIGGER:
                self._running = False
                self.prop_dict['finished'][0] = PVM_Create.PVM_FLAG_VAL_RESET
            if self.prop_dict['finished'][0] == PVM_Create.PVM_FLAG_TRIGGER_DISPLAY:
                self.do_display = not self.do_display
                if not self.do_display:
                    cv2.destroyWindow("PVM display")
                    cv2.destroyAllWindows()
                    for i in range(10):
                        cv2.waitKey(1)
                self.prop_dict['finished'][0] = PVM_Create.PVM_FLAG_VAL_RESET
            if self.prop_dict['finished'][0] == PVM_Create.PVM_FLAG_TRIGGER_RECORD:
                self.record = not self.record
                if self.record:
                    self.do_display = True
                    self.video_recorder.set_filename(self.prop_dict["record_filename"].tostring())
                elif self.video_recorder is not None:
                    self.video_recorder.finish()
                self.prop_dict['finished'][0] = 0

    def process_dream_experiment(self):
        if self.signal.get_index() == self.prop_dict['flags'][2]:
            if "stage0" not in self.dream_experiment_data.keys():
                self.dream_experiment_data["stage0"] = True
                logging.info("Dream experiment stage0 has begun")
            elif "stage1" not in self.dream_experiment_data.keys():
                self.dream_experiment_data["stage0"] = False
                self.dream_experiment_data["stage1"] = True
                self.prop_dict['flags'][0] = 1  # begin dreaming
                logging.info("Dream experiment stage1 has begun")
                self.freeze_learning()
                for (i, k) in enumerate(self.prop_dict["learning_rates"]):
                    k[0] = -0.00001
            else:
                self.dream_experiment = False
                self.prop_dict['flags'][0] = 0  # end dreaming
                CoreUtils.save_model(self.dream_experiment_data, "dream_data.p.gz")
                self.dream_experiment_data = {}
                logging.info("Dream experiment has ended")
                self.un_freeze_learning()
        # Stage 0 is ongoing.
        if "stage0" in self.dream_experiment_data.keys() and self.dream_experiment_data["stage0"] is True:
            if "stage0_data" not in self.dream_experiment_data.keys():
                self.dream_experiment_data['stage0_data'] = []
            self.dream_experiment_data['stage0_data'].append((self.actual_input_prev.copy(), self.predicted.copy()))
        # Stage 0 is ongoing.
        if "stage1" in self.dream_experiment_data.keys() and self.dream_experiment_data["stage1"] is True:
            if "stage1_data" not in self.dream_experiment_data.keys():
                self.dream_experiment_data['stage1_data'] = []
            self.dream_experiment_data['stage1_data'].append(self.predicted.copy())

    def running(self):
        """
        While returning True the simulation will keep going.
        """
        return self._running

    def finish(self):
        if self.collect_error:
            f = open("predictive_error.txt", "w")
            print >>f, "%4.4f" % self.predictive_error
            f.close()
            f = open("predictive_error_array.txt", "w")
            for p in self.predictive_error_array:
                print >>f, "%4.4f" % p
            f.close()
            f = open("reconstruction_error.txt", "w")
            print >>f, "%4.4f" % self.reconstruction_error
            f.close()
            f = open("reconstruction_error_array.txt", "w")
            for p in self.reconstruction_error_array:
                print >>f, "%4.4f" % p
            f.close()
        if self.video_recorder is not None:
            self.video_recorder.finish()
        self.signal.finish()

    def freeze_learning(self):
        self.freeze_learning_scheduled = True

    def un_freeze_learning(self):
        self.unfreeze_learning_scheduled = True

    def freeze_learning_synchronised(self):
        """
        Saves the current learning rates of the model in a temporary buffer
        :return:
        """
        self.freeze_learning_scheduled = False
        if not self.frozen_learning:
            self.frozen_learning = True
            self.learning_rate_buffer = []
            self.readout_learning_rate_buffer = []
            for k in range(len(self.prop_dict['learning_rates'])):
                self.learning_rate_buffer.append(self.prop_dict['learning_rates'][k].copy())
                self.prop_dict['learning_rates'][k][0] = 0.0
            self.readout_learning_rate_buffer.append(self.prop_dict["readout_learning_rate"].copy())
            self.prop_dict["readout_learning_rate"][0] = 0.0
            logging.info("Freezing model learning")

    def un_freeze_learning_synchronised(self):
        """
        Recovers the previously stored learning rates.
        :return:
        """
        self.unfreeze_learning_scheduled = False
        if self.frozen_learning:
            for (i, k) in enumerate(self.prop_dict["learning_rates"]):
                k[0] = self.learning_rate_buffer[i][0]
            self.prop_dict["readout_learning_rate"][0] = self.readout_learning_rate_buffer[0][0]
            self.frozen_learning = False
            logging.info("Unfreezing model learning")
