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
import numpy as np
import cv2
import logging
import os


class VideoRecorder(object):
    def __init__(self, rec_filename):
        """
        :param rec_filename:
        :return:

        Handy object to carry out video recording
        """
        if rec_filename[-4:] == ".avi":
            rec_filename = rec_filename[:-4]
        rec_filename = rec_filename + "_%02d.avi"
        self.rec_filename = rec_filename
        self._video = None
        self.index = 0

    def set_filename(self, filename):
        self.override_name = filename

    def _get_filename(self):
        if self.override_name is None:
            return self.rec_filename % self.index
        else:
            return self.override_name

    def record(self, image):
        """
        :param image:
        :return:

        Takes and image and records it into a file
        """
        if self._video is None:
                self._video = cv2.VideoWriter()
                fps = 20
                retval = self._video.open(os.path.expanduser(self._get_filename()),
                                          cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),
                                          fps, (image.shape[1], image.shape[0]))
                assert(retval)
                logging.info("Creating an avi file %s" % os.path.expanduser(self._get_filename()))
        self._video.write(image)

    def finish(self):
        """
        When done releases the cv video writer
        :return:
        """
        if self._video is not None:
            self._video.release()
            self._video = None
            self.index += 1
            logging.info("Finished recording file %s" % os.path.expanduser(self._get_filename()))


class DisplayHelperObject(object):
    """
    An object that allows to easily assemble display frame
    """
    def __init__(self, width=1920, height=1080, margin=8, grid=(4, 5)):
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.margin = margin
        self.im_size = min((width-2*self.margin)/grid[0], (height-2*self.margin)/grid[1])-2*self.margin
        self.grid = grid
        self.clear_frame()
        self.logo = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bc_logo_gray_sm.png"))
        self.no_refresh = {}

    def grid_to_pix(self, x, y):
        if x < 0 or x > self.grid[0] or y < 0 or y > self.grid[1]:
            raise Exception("Out of grid!")
        px = x*(self.im_size+2*self.margin)+2*self.margin
        py = y*(self.im_size+2*self.margin)+2*self.margin
        return py, px
    
    def place_gray_image(self, grid_x, grid_y, image, text=""):
        (px, py) = self.grid_to_pix(grid_x, grid_y)
        self.place_label(grid_x=grid_x, grid_y=grid_y, text=text)
        resized = cv2.resize(image, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_NEAREST)
        self.frame[px:px+self.im_size, py:py+self.im_size, 0] = resized
        self.frame[px:px+self.im_size, py:py+self.im_size, 1] = resized
        self.frame[px:px+self.im_size, py:py+self.im_size, 2] = resized
        cv2.rectangle(self.frame, (py-2, px-2), (py+self.im_size+1, px+self.im_size+1), color=(140, 100, 100))

    def place_gray_float_image(self, grid_x, grid_y, image, text=""):
        (px, py) = self.grid_to_pix(grid_x, grid_y)
        self.place_label(grid_x=grid_x, grid_y=grid_y, text=text)
        resized = cv2.resize(image, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_NEAREST)
        resized *= 255
        resized = resized.astype(np.uint8)
        self.frame[px:px+self.im_size, py:py+self.im_size, 0] = resized
        self.frame[px:px+self.im_size, py:py+self.im_size, 1] = resized
        self.frame[px:px+self.im_size, py:py+self.im_size, 2] = resized
        cv2.rectangle(self.frame, (py-2, px-2), (py+self.im_size+1, px+self.im_size+1), color=(140, 100, 100))

    def place_color_image(self, grid_x, grid_y, image, text=""):
        (px, py) = self.grid_to_pix(grid_x, grid_y)
        self.place_label(grid_x=grid_x, grid_y=grid_y, text=text)
        cv2.rectangle(self.frame, (py-2, px-2), (py+self.im_size+1, px+self.im_size+1), color=(140, 100, 100))
        self.frame[px:px+self.im_size, py:py+self.im_size, :] = cv2.resize(image, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_NEAREST)

    def place_color_float_image(self, grid_x, grid_y, image, text=""):
        (px, py) = self.grid_to_pix(grid_x, grid_y)
        self.place_label(grid_x=grid_x, grid_y=grid_y, text=text)
        cv2.rectangle(self.frame, (py-2, px-2), (py+self.im_size+1, px+self.im_size+1), color=(140, 100, 100))
        resized = cv2.resize(image, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_NEAREST)
        resized *= 255
        resized = resized.astype(np.uint8)
        self.frame[px:px+self.im_size, py:py+self.im_size, :] = resized

    def place_image(self, grid_x, grid_y, image, text=""):
        if image.dtype == np.float:
            if len(image.shape) == 3 and image.shape[2] == 3:
                self.place_color_float_image(grid_x, grid_y, image, text)
            else:
                self.place_gray_float_image(grid_x, grid_y, image, text)
        else:
            if len(image.shape) == 3 and image.shape[2] == 3:
                self.place_color_image(grid_x, grid_y, image, text)
            else:
                self.place_gray_image(grid_x, grid_y, image, text)

    def place_color_logo(self, grid_x, grid_y, image=None, text=""):
        if image is None:
            image = self.logo
        (px, py) = self.grid_to_pix(grid_x, grid_y)
        self.frame[px:px+image.shape[0], py:py+image.shape[1], :] = image

    def place_label(self, grid_x, grid_y, text=""):
        if not (grid_x, grid_y) in self.no_refresh:
            (px, py) = self.grid_to_pix(grid_x, grid_y)
            cv2.putText(self.frame, text, (py+5, px-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 0), lineType=cv2.CV_AA)
            self.no_refresh[(grid_x, grid_y)] = 0

    def place_text(self, grid_x, grid_y, voffset=0, text=""):
        (px, py) = self.grid_to_pix(grid_x, grid_y)
        cv2.putText(self.frame, text, (py, px + voffset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 0), lineType=cv2.CV_AA)

    def clear_cell(self, grid_x, grid_y):
        (px, py) = self.grid_to_pix(grid_x, grid_y)
        self.frame[px:px+self.im_size, py:py+self.im_size, :] = 127

    def clear_frame(self):
        self.frame *= 0
        self.frame += 127
        self.no_refresh = {}
