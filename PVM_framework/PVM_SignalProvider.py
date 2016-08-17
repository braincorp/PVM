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
import PVM_framework.AbstractExecutionManager as AbstractExecutionManager
import logging
from PVM_tools.labeled_movie import FrameCollection
import cv2
import os
import numpy as np


class SimpleSignalProvider(AbstractExecutionManager.AbstractSignalProvider):
    def __init__(self, frame_resolution, heatmap_resolution, files, storage, channel="default", remove_files=True, reverse=False):
        """
        Handy object to provide input frames and supervising heatmaps efficiently

        :param frame_resolution:
        :param heatmap_resolution:
        :param files:
        :param channel:
        """
        self.files = files
        self.frame_resolution = frame_resolution
        self.heatmap_resolution = heatmap_resolution
        self.channel = channel
        self.frames = []
        self.nframes = 0
        self.masks = []
        self.index = 0
        self.remove_files = remove_files
        self.reverse = reverse
        self.storage=storage

    def start(self):
        """
        Collect and preprocess all the data
        :return:
        """
        for (labeled_file, channel) in self.files:
            logging.info("Incorporating file %s" % (labeled_file))
            local_labeled_file = self.storage.get(labeled_file)
            if local_labeled_file is not None:
                self.fc = FrameCollection()
                self.fc.load_from_file(local_labeled_file)
                self.nframes += len(self.fc)
                for i in xrange(len(self.fc)):
                    img1 = self.fc.Frame(i).get_image(channel=channel)
                    img = cv2.resize(img1, dsize=self.frame_resolution, interpolation=cv2.INTER_CUBIC)
                    label = self.fc.Frame(i).get_label(channel=channel)
                    if label is not None:
                        label.set_image_shape(shape=img1.shape)
                        mask = self.fc.Frame(i).get_label(channel=channel).get_mask()
                        mask = cv2.resize(mask, dsize=self.heatmap_resolution, interpolation=cv2.INTER_CUBIC)
                    else:
                        mask = np.zeros(self.heatmap_resolution, dtype=np.uint8)
                    self.masks.append(mask)
                    self.frames.append(img)
                if self.remove_files:
                    os.remove(local_labeled_file)

    def get_signal(self, name, time):
        """
        Return the requested signal. If time is zero, return the current signal, otherwise
        return past of future signals as indicated by the time parameter

        :param name:
        :param time:
        :return:
        """
        if name == "frame":
            return self.frames[(self.index+time) % self.nframes]
        elif name == "mask":
            return self.masks[(self.index+time) % self.nframes]
        else:
            raise Exception("Unknown signal type")

    def get_length(self):
        return self.nframes

    def advance(self):
        """
        Move current index by one step forward
        :return:
        """
        if self.reverse:
            self.index = (self.index - 1) % self.nframes
        else:
            self.index = (self.index + 1) % self.nframes

    def finish(self):
        """
        Cleanup
        :return:
        """
        pass

    def reset(self):
        self.index = 0

    def get_index(self):
        """
        Return the index of the current frame in the sequence
        :return:
        """
        return self.index


class StereoSignalProvider(AbstractExecutionManager.AbstractSignalProvider):
    def __init__(self, frame_resolution, heatmap_resolution, files, storage, channel="default", remove_files=True, only_one_channel=""):
        """
        Handy object to provide input frames and supervising heatmaps efficiently

        :param frame_resolution:
        :param heatmap_resolution:
        :param files:
        :param channel:
        :return:
        """
        self.files = files
        self.frame_resolution = frame_resolution
        self.frame_resolution_half = (frame_resolution[0], frame_resolution[1]/2)

        self.heatmap_resolution = heatmap_resolution
        self.channel = channel
        self.frames = []
        self.nframes = 0
        self.masks = []
        self.index = 0
        self.remove_files = remove_files
        self.only_one_channel = only_one_channel
        self.storage=storage

    def start(self):
        """
        Collect and preprocess all the data
        :return:
        """
        for (labeled_file, channel) in self.files:
            logging.info("Incorporating file %s" % (labeled_file))
            local_labeled_file = self.storage.get(labeled_file)
            if local_labeled_file is not None:
                self.fc = FrameCollection()
                self.fc.load_from_file(local_labeled_file)
                self.nframes += len(self.fc)
                for i in xrange(len(self.fc)):
                    img1 = self.fc.Frame(i).get_image(channel="left")
                    imgl = cv2.resize(img1, dsize=self.frame_resolution_half, interpolation=cv2.INTER_CUBIC)
                    img1 = self.fc.Frame(i).get_image(channel="right")
                    imgr = cv2.resize(img1, dsize=self.frame_resolution_half, interpolation=cv2.INTER_CUBIC)
                    if self.only_one_channel == "left":
                        imgr *= 0
                    if self.only_one_channel == "right":
                        imgl *= 0
                    img = self.interlace_two_images(imgl, imgr)
                    label = self.fc.Frame(i).get_label(channel="left")
                    if label is not None:
                        label.set_image_shape(shape=img1.shape)
                        mask = label.get_mask()
                        mask = cv2.resize(mask, dsize=self.heatmap_resolution, interpolation=cv2.INTER_CUBIC)
                    else:
                        mask = np.zeros(self.heatmap_resolution, dtype=np.uint8)
                    self.masks.append(mask)
                    self.frames.append(img)
                if self.remove_files:
                    os.remove(local_labeled_file)

    def interlace_two_images(self, img1, img2):
        interlacedimg = np.zeros((img1.shape[0]+img2.shape[0],
                                  img1.shape[1], img1.shape[2]), dtype=np.uint8)
        interlacedimg[0::2, :, :] = img1
        interlacedimg[1::2, :, :] = img2
        return interlacedimg

    def get_signal(self, name, time):
        """
        Return the requested signal. If time is zero, return the current signal, otherwise
        return past of future signals as indicated by the time parameter

        :param name:
        :param time:
        :return:
        """
        if name == "frame":
            return self.frames[(self.index+time) % self.nframes]
        elif name == "mask":
            return self.masks[(self.index+time) % self.nframes]
        else:
            raise Exception("Unknown signal type")

    def get_length(self):
        return self.nframes

    def advance(self):
        """
        Move current index by one step forward
        :return:
        """
        self.index = (self.index + 1) % self.nframes

    def finish(self):
        """
        Cleanup
        :return:
        """
        pass

    def reset(self):
        self.index = 0

    def get_index(self):
        """
        Return the index of the current frame in the sequence
        :return:
        """
        return self.index


class TripleSignalProvider(AbstractExecutionManager.AbstractSignalProvider):
    def __init__(self, frame_resolution, heatmap_resolution, files1, files2, files3, storage, channel="default", remove_files=True):
        """
        Handy object to provide input frames and supervising heatmaps efficiently

        :param frame_resolution:
        :param heatmap_resolution:
        :param files:
        :param channel:
        :return:
        """
        self.files1 = files1
        self.files2 = files2
        self.files3 = files3
        self.frame_resolution = frame_resolution
        self.heatmap_resolution = heatmap_resolution
        self.channel = channel
        self.frames = []
        self.nframes = 0
        self.masks = []
        self.index = 0
        self.remove_files = remove_files
        self.storage = storage

    def buffer_files(self, idx, channel, files):
        for (labeled_file, channel) in files:
            logging.info("Incorporating file %s" % (labeled_file))
            local_labeled_file = self.storage.get(labeled_file)
            if local_labeled_file is not None:
                self.fc = FrameCollection()
                self.fc.load_from_file(local_labeled_file)
                self.nframes += len(self.fc)
                for i in xrange(len(self.fc)):
                    img1 = self.fc.Frame(i).get_image(channel=channel)
                    img = cv2.resize(img1, dsize=self.frame_resolution, interpolation=cv2.INTER_CUBIC)
                    label = self.fc.Frame(i).get_label(channel=channel)
                    label.set_image_shape(shape=img1.shape)
                    mask = self.fc.Frame(i).get_label(channel=channel).get_mask()
                    mask = cv2.resize(mask, dsize=self.heatmap_resolution, interpolation=cv2.INTER_CUBIC)
                    mask3d = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=mask.dtype)
                    mask3d[:, :, idx] = mask
                    self.masks.append(mask3d)
                    self.frames.append(img)
                if self.remove_files:
                    os.remove(local_labeled_file)

    def start(self):
        """
        Collect and preprocess all the data
        :return:
        """
        self.buffer_files(0, channel=self.channel, files=self.files1)
        self.buffer_files(1, channel=self.channel, files=self.files2)
        self.buffer_files(2, channel=self.channel, files=self.files3)

    def get_signal(self, name, time):
        """
        Return the requested signal. If time is zero, return the current signal, otherwise
        return past of future signals as indicated by the time parameter

        :param name:
        :param time:
        :return:
        """
        if name == "frame":
            return self.frames[(self.index+time) % self.nframes]
        elif name == "mask":
            return self.masks[(self.index+time) % self.nframes]
        else:
            raise Exception("Unknown signal type")

    def get_length(self):
        return self.nframes

    def advance(self):
        """
        Move current index by one step forward
        :return:
        """
        self.index = (self.index + 1) % self.nframes

    def finish(self):
        """
        Cleanup
        :return:
        """
        pass

    def reset(self):
        self.index = 0

    def get_index(self):
        """
        Return the index of the current frame in the sequence
        :return:
        """
        return self.index
