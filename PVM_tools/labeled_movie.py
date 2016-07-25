"""
This module contains a number of classes representing handling, loading and encoding a movie object with labels.

The classes included are:

    1. LabeledMovieFrame - the primary container for images, labels and other data
    2. LabeledMovieHeader - header object used in writing files
    3. LabeledMovieReader - Reader object for dealing with files
    4. LabeledMovieWriter - Writer object for dealing with files
    5. FrameCollecion - a container object representing a movie when it is loaded in memory

In general in most of the cases it will be sufficient to load the FrameCollecion object into any project dealing
with labeled movies, as it has methods for loading, saving and accessing frames.

Import this module as:
::

    import PVM_tools.labeled_movie

or:
::

    from PVM_tools.labeled_movie import LabeledMovieFrame, LabeledMovieHeader, \\
        LabeledMovieReader, LabeledMovieWriter, FrameCollection

"""
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
import PVM_framework.CoreUtils as CoreUtils
import cv2
import numpy as np
import time


class LabeledMovieFrame(object):
    """
    :param internal_storage_method: Storage method 'raw'|'png'|'jpg'
    :type internal_storage_method: str
    :param compression_level: A number 0-100 describing the level of compression. Lower is more compressed.
    :type compression_level: int

    A generic class representing a movie frame.

    As frame is an object composed of multiple elements:

        * One or more channels
        * One or more images (each assigned to a channel)
        * One or more bounding regions (labels) each assigned to a channel
        * One or more time stamps, each assigned to a channel
        * One or more metadata strings, each ssigned to a channel

    There always exist a default channel in a frame. That default channel can be redirected
    to any other channel created by the user. E.g. there may exist a frame which has two images
    (e.g left and right taken from a stereo camera). The default channel will point to one of them
    e.g. left.

    The internal storage algorithm for images can be selected from:

        * 'raw' - no compression, raw representation
        * 'png' - png compression algorithm
        * 'jpg' - JPEG compression algorithm (loses information)

    jpg is recommended for long and/or large movies, for practical reasons. It can save ~10x the space, but the image
    will lose some quality.

    :Example:
    ::

        cap = cv2.VideoCapture(-1)
        ret, img = cap.read()
        F = LabeledMovieFrame(internal_storage_method="jpg")
        F.set_image(img)


    """

    def __init__(self, internal_storage_method="raw", compression_level=80):
        self._version = 0.4
        if internal_storage_method not in ["raw", "png", "jpg"]:
            raise "Unsupported storage method"
        self._internal_storage_method = internal_storage_method
        self._per_channel_storage_method = {}
        self._image_storage = {}
        self._audio_storage = {}
        self._label_storage = {}
        self._timestamp_storage = {}
        self._metadata_storage = {}
        self._compression_level = np.clip(compression_level, 0, 99)
        self._channel = {"default": "default"}

    def create_channel(self, channel):
        """
        :param channel: Name of the channel to create
        :type channel: str

        Creates a new channel. A channel is a container which can take an image and other attributes.

        :Example:
        ::

            cap0 = cv2.VideoCapture(0)
            cap1 = cv2.VideoCapture(1)
            ret, img_left = cap0.read()
            ret, img_right = cap1.read()
            F = LabeledMovieFrame(internal_storage_method="jpg")
            F.create_channel("left_image")
            F.set_image(img_left, channel="left_image")
            F.create_channel("right_image")
            F.set_image(img_right, channel="right_image")
            F.set_default_channel("left_image")
            assert(np.allclose(F.get_image(), img_left))

        """
        self._channel[channel] = channel
        self._per_channel_storage_method[channel] = self._internal_storage_method

    def set_default_channel(self, channel, rename_previous_default="channel01"):
        """
        :param channel: Name of the channel to create
        :type channel: str
        :param rename_previous_default: Optional new name of the previous default channel
        :type channel: str

        Set the default channel. A channel is a container which can take an image and other attributes.
        A default channel is the one that will be used if the channel parameter is omitted.

        In certain cases only the default channel existed in the frame, new channel got created and is set to
        default. In this case the previous content of the default channel could become inaccessible. In such
        case the previous default channel will be renamed with 'rename_previous_default' name (default 'channel01').

        :Example:
        ::

            cap0 = cv2.VideoCapture(0)
            cap1 = cv2.VideoCapture(1)
            ret, img_left = cap0.read()
            ret, img_right = cap1.read()
            F = LabeledMovieFrame(internal_storage_method="jpg")
            F.create_channel("left_image")
            F.set_default_channel("left_image")
            F.set_image(img_left) # Now left_image is the default channel
            F.create_channel("right_image")
            F.set_image(img_right, channel="right_image")
            assert(np.allclose(F.get_image(), img_left))

        """
        if "default" in self._channel.keys():
            self._channel[rename_previous_default] = self._channel["default"]
        self._channel["default"] = channel

    def set_image(self, image, channel="default", storage_method=None):
        """
        :param image: numpy array containing an RGB representation of an image
        :type image: numpy.ndarray
        :param channel: optional argument defining with which channel the image should be associated (default: "default")
        :param channel: str
        :return: Integer indicating whether the processing of the image was successful
        :rtype: int

        Inserts an image into a channel.
        """
        ret = 0
        if storage_method is None:
            self._per_channel_storage_method[channel] = self._internal_storage_method
        elif storage_method in ["raw", "png", "jpg"]:
            self._per_channel_storage_method[channel] = storage_method
        else:
            raise Exception("Unknown storage method")
        if self._per_channel_storage_method[channel] == "raw":
            self._image_storage[self._channel[channel]] = image.copy()
            ret = 0
        if self._per_channel_storage_method[channel] == "png":
            quality = 9 - int(self._compression_level/10)
            (ret, buf) = cv2.imencode(".png", image, (cv2.IMWRITE_PNG_COMPRESSION, quality))
            self._image_storage[self._channel[channel]] = buf
        if self._per_channel_storage_method[channel] == "jpg":
            (ret, buf) = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, self._compression_level))
            self._image_storage[self._channel[channel]] = buf
        return ret

    def set_label(self, label, channel="default", target="default"):
        """
        :param label: A serializable object containing information about the frame. E.g. PVM_tools.BoundingRegion
        :type label: object
        :param channel: optional argument defining with which channel the image should be associated (default: "default")
        :type channel: str
        :param target: name of the target (there may be many labels/targets on a single image)
        :type target: str

        Inserts an label into a channel.
        """
        if self._channel[channel] not in self._label_storage.keys():
            self._label_storage[self._channel[channel]] = {}
        self._label_storage[self._channel[channel]][target] = label

    def set_timestamp(self, timestamp, channel="default"):
        """
        :param timestamp: A serializable object containing time information. E.g. time as float.
        :type timestamp: object
        :param channel: optional argument defining with which channel the image should be associated (default: "default")
        :param channel: str

        Inserts an timestamp into a channel.
        """
        self._timestamp_storage[self._channel[channel]] = timestamp

    def set_metadata(self, metadata, channel="default"):
        """
        :param metadata: A serializable object containing metadata information. E.g. a string.
        :type metadata: object
        :param channel: optional argument defining with which channel the image should be associated (default: "default")
        :param channel: str

        Inserts a metadata into a channel.
        """
        self._metadata_storage[self._channel[channel]] = metadata

    def set_audio(self, audio, channel="default"):
        """
        :param audio: A serializable object containing audio information.
        :type audio: object
        :param channel: optional argument defining with which channel the image should be associated (default: "default")
        :param channel: str

        Inserts an audio information into a channel.
        """
        self._audio_storage[self._channel[channel]] = audio

    def get_channels(self):
        """
        Retuns the names of channels available

        :return: list of channels
        :rtype: list
        """
        return self._channel.keys()

    def get_image(self, channel="default"):
        """
        :param channel: optional argument selecting the channel (will use default channel if omitted)
        :type channel: str
        :return: decoded image in RGB format
        :rtype: numpy.ndarray

        Decodes and returns the image from the given channel
        """
        if self._per_channel_storage_method[self._channel[channel]] == "raw":
            return self._image_storage[self._channel[channel]]
        if self._per_channel_storage_method[self._channel[channel]] == "png":
            im_data = self._image_storage[self._channel[channel]]
            return cv2.imdecode(im_data, cv2.CV_LOAD_IMAGE_COLOR)
        if self._per_channel_storage_method[self._channel[channel]] == "jpg":
            im_data = self._image_storage[self._channel[channel]]
            return cv2.imdecode(im_data, cv2.CV_LOAD_IMAGE_COLOR)

    def get_label(self, channel="default", target="default"):
        """
        :param channel: optional argument selecting the channel (will use default channel if omitted)
        :type channel: str
        :return: label object
        :rtype: object

        Decodes and returns the label object from the given channel. Returns None if no label is present.
        """
        try:
            return self._label_storage[self._channel[channel]][target]
        except KeyError:
            return None

    def get_targets(self, channel="default"):
        """
        :param channel: identifier of the channel
        :type channel: str
        :return: list of target labels
        :type: list

        Returns the list of target names.
        """
        try:
            return self._label_storage[self._channel[channel]].keys()
        except:
            return None

    def get_audio(self, channel="default"):
        """
        :param channel: optional argument selecting the channel (will use default channel if omitted)
        :type channel: str
        :return: audio object
        :rtype: object

        Decodes and returns the audio object from the given channel
        """
        try:
            return self._audio_storage[self._channel[channel]]
        except KeyError:
            return None

    def get_timestamp(self, channel="default"):
        """
        :param channel: optional argument selecting the channel (will use default channel if omitted)
        :type channel: str
        :return: timestamp object
        :rtype: object

        Decodes and returns the timestamp object from the given channel
        """
        try:
            return self._label_storage[self._channel[channel]]
        except KeyError:
            return None

    def get_metadata(self, channel="default"):
        """
        :param channel: optional argument selecting the channel (will use default channel if omitted)
        :type channel: str
        :return: metadata object
        :rtype: object

        Decodes and returns the metadata object from the given channel
        """
        try:
            return self._metadata_storage[self._channel[channel]]
        except KeyError:
            return None

    @property
    def version(self):
        return self._version

    @property
    def storage_method(self):
        if self._version < 0.3:
            return self.internal_storage_method
        else:
            return self._internal_storage_method

    @property
    def compression(self):
        if self._version < 0.3:
            return self.compression_level
        else:
            return self._compression_level

    @classmethod
    def upgrade_to_latest_version(cls, Frame):
        if Frame.version < 0.3:
            storage_method = Frame.internal_storage_method
            compression = Frame.compression_level
            New_frame = LabeledMovieFrame(internal_storage_method=storage_method, compression_level=compression)
            channels = Frame.channel.keys()
            for channel in channels:
                New_frame.set_image(image=Frame.get_image(channel=channel), channel=channel)
                try:
                    label = Frame._label_storage[channel]
                except KeyError:
                    label = None
                if label is not None:
                    New_frame.set_label(label, channel=channel)
                audio = Frame.get_audio(channel=channel)
                if audio is not None:
                    New_frame.set_audio(audio, channel=channel)
            return New_frame
        elif Frame.version < 0.4:
            storage_method = Frame.storage_method
            compression = Frame.compression
            New_frame = LabeledMovieFrame(internal_storage_method=storage_method, compression_level=compression)
            channels = Frame.get_channels()
            for channel in channels:
                New_frame.set_image(image=Frame.get_image(channel=channel), channel=channel)
                label = Frame.get_label(channel=channel)
                if label is not None:
                    New_frame.set_label(label, channel=channel)
                audio = Frame.get_audio(channel=channel)
                if audio is not None:
                    New_frame.set_audio(audio, channel=channel)
            return New_frame
        else:
            return Frame

    def __setstate__(self, state):
        self.__dict__ = state
        if self._version < 0.3:
            self._internal_storage_method = self.internal_storage_method
            self._image_storage = self.image_storage
            self._audio_storage = self.audio_storage
            self._label_storage = self.label_storage
            self._channel = self.channel
        if self._version < 0.4:
            self._per_channel_storage_method={}
            for channel in self._channel.keys():
                self._per_channel_storage_method[channel]=self._internal_storage_method


class LabeledMovieHeader(object):
    """
    :param fps: frames per second, parameter defining the sampling rate of the movie necessary for proper playback
    :type fps: float
    :param created_date: date of creation as returned by time.ctime()
    :type created_date: str
    :param author: Author information
    :type author: str
    :param copyright: Copyright information
    :type copyright: str

    A header object containing additional information about the collection
    of movie frames stored in a file.
    """
    def __init__(self, fps=20.0, created_date=time.ctime(), author="", copyright="2015 Brain Corporation"):
        self._fps = fps
        self._version = 0.1
        self._created_date = created_date
        self._author = author
        self._copyright = copyright

    @property
    def fps(self):
        """
        Get the frame per second parameter

        :return: fps
        :rtype: float
        """
        return self._fps

    @property
    def version(self):
        """
        Get the version parameter

        :return: version
        :rtype: float
        """
        return self._version

    @property
    def created_date(self):
        """
        Get the creation date parameter

        :return: date as returned by time.ctime()
        :rtype: str
        """
        return self._created_date

    @property
    def author(self):
        """
        Get the author information

        :return: Author information string
        :rtype: str
        """
        return self._author

    @property
    def copyright(self):
        """
        Get the copyright information
        :return: copyright
        :rtype: str
        """
        return self._copyright


class LabeledMovieWriter(object):
    """
    :param filename: name of the file to written. Caution, if file exists it will be overwritten
    :type filename: str
    :param movie_header: Additional movie information header
    :type movie_header: PVM_tools.labeled_movie.LabeledMovieHeader

    A helper class allowing to write frames to a file. If the movie_header argument is not given it will
    generate a default LabeledMovieHeader object.

    :Example:
    ::

        cap = cv2.VideoCapture(-1)
        wr = LabeledMovieWriter("./movie_test.pkl")
        for i in xrange(100):
            ret, img = cap.read()
            F = LabeledMovieFrame(internal_storage_method="jpg")
            F.set_image(img)
            wr.write_frame(F)
        wr.finish()

    """
    def __init__(self, filename, movie_header=LabeledMovieHeader()):
        self.file = open(filename, mode="wb")
        self._movie_header = movie_header
        self._movie_header_written = False

    def write_frame(self, Frame):
        """
        :param Frame: a frame to be written
        :type Frame: PVM_tools.labeled_movie.LabeledMovieFrame
        """
        if not self._movie_header_written:
            cPickle.dump(obj=self._movie_header, file=self.file, protocol=-1)
            self._movie_header_written = True
        cPickle.dump(obj=Frame, file=self.file, protocol=-1)

    def finish(self):
        """
        Finish a session and close the file.
        """
        self.file.close()


class LabeledMovieReader(object):
    """
    :param filename: path to the file
    :type filename: str

    A helper class for reading a collection of frames from a file.
    """
    def __init__(self, filename):
        self.file = open(filename, mode="rb")
        self._has_frames = True
        try:
            self._header = CoreUtils.load_legacy_pickle_file(self.file)
        except EOFError:
            self._has_frames = False

    @property
    def fps(self):
        """
        Property returning the frame per second parameter from the file header

        :return:
        """
        if self._has_frames:
            return self._header.fps
        else:
            return None

    def frames(self):
        """
        A generator call for returning the collection of frames.

        :Example:
        ::

            mr = LabeledMovieReader("./movie_test.pkl")
            for Frame in mr.frames():
                img = Frame.get_image()
                cv2.imshow("Image", img)
                cv2.waitKey(int(1000/mr.fps))

        :return: a frame iterator
        """
        ReferenceFrame = LabeledMovieFrame()
        ver = ReferenceFrame.version
        warning_message = True
        while True:
            try:
                F = CoreUtils.load_legacy_pickle_file(self.file)
                if F.version < ver:
                    if warning_message:
                        print "This movie uses old version frames (%2.2f), needs upgrading" % F.version
                        print "Performance will be affected!"
                        warning_message = False
                    F = LabeledMovieFrame.upgrade_to_latest_version(F)
                yield F
            except EOFError:
                self._has_frames = False
                break

    def get_header(self):
        """
        Returns the entire header object
        """
        return self._header


class FrameCollection(object):
    """
    A helper class encapsulating some of the details of working with a collection of frames.

    Encapsulates the movie header object, a list of frames and provides convenient read and write mechanisms.

    :Example:
    ::

        fc = FrameCollection()
        for i in xrange(100):
            ret, img = cap.read()
            F = LabeledMovieFrame(internal_storage_method="jpg")
            F.set_image(img)
            fc.append(F)

        fc.reverse()
        for image in fc:
            cv2.imshow("frame", image)
            cv2.waitKey(int(1000/fc.fps))



    """
    def __init__(self, channel="default", movie_header=LabeledMovieHeader(fps=20)):
        self._list = []
        self._channel = channel
        self._header = movie_header

    def append(self, Frame):
        """
        Add a frame to the collection

        :param Frame: frame object
        :type Frame: PVM_tools.labeled_movie.LabeledMovieFrame
        """
        self._list.append(Frame)

    def set_active_channel(self, channel):
        """
        :param channel: channel to select
        :type channel: str
        :return:

        Selects the active channel from which image data can be conveniently extracted by list interface
        """
        self._channel = channel

    def delete(self, Frame):
        """
        :param Frame: index of, or the frame object itself to be removed
        :type Frame: int|PVM_tools.labeled_movie.LabeledMovieFrame

        Removes a frame from the collection
        """
        if type(Frame) == int:
            self._list.pop(Frame)
        elif type(Frame) == LabeledMovieFrame:
            self._list.remove(Frame)

    def __getitem__(self, item):
        self._list[item].get_image(channel=self._channel)

    def __iter__(self):
        for F in self._list:
            yield F.get_image(channel=self._channel)

    def __len__(self):
        return len(self._list)

    def reverse(self):
        """
        Reverses the order of frames

        :return:
        """
        self._list.reverse()

    @property
    def fps(self):
        """
        Get the frame per second property

        :return:
        """
        return self._header.fps

    def write_to_file(self, filename):
        """
        Use a LabeledMovieWriter class to write the entire collection to a file.

        :param filename: path to the file written
        :type filename: str
        """
        mw = LabeledMovieWriter(filename=filename, movie_header=self._header)
        for Frame in self._list:
            mw.write_frame(Frame)
        mw.finish()

    def load_from_file(self, filename):
        """
        :param filename: path to the file read
        :type filename: str

        Load a collection from a file using the LabeledMovieReader class
        """
        mr = LabeledMovieReader(filename=filename)
        self._header = mr.get_header()
        self._list = []
        for Frame in mr.frames():
            self._list.append(Frame)

    def Frame(self, index):
        """
        :param index: Index of the frame to be exposed
        :type index: int
        :return: PVM_tools.labeled_movie.LabeledMovieFrame

        Exposes the frame in the collection at a given index.
        """
        return self._list[index]


if __name__ == "__main__":
    cap = cv2.VideoCapture(-1)
    wr = LabeledMovieWriter("./movie_test.pkl")
    fc = FrameCollection()
    for i in xrange(100):
        ret, img = cap.read()
        F = LabeledMovieFrame(internal_storage_method="jpg")
        F.set_image(img)
        wr.write_frame(F)
        fc.append(F)
    wr.finish()

    fc.reverse()
    for image in fc:
        cv2.imshow("frame", image)
        cv2.waitKey(int(1000/fc.fps))

    mr = LabeledMovieReader("./movie_test.pkl")
    for Frame in mr.frames():
        img = Frame.get_image()
        cv2.imshow("Image", img)
        cv2.waitKey(int(1000/mr.fps))
