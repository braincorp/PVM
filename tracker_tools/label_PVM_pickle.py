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
"""
Labeling movies
########

The label_PVM_pickle.py script is an easy to use tool for attaching labels to movies in
labeled movie format. The process is easy and is executed in the following steps.

    1. First identify a movie file in a regular format (e.g. mov, avi). Next use a conversion tool
        to convert it into a pickle. E.g. if the file was called example.mov one could run:

        ::

            python movie_to_PVM_pickle.py example.mov -o example.pkl

        This tool allows for additional processing, e.g. rotations, flipping, scaling. Refer to
        the documentation.

    2. Once the pickle has been created, it can be tagged (labeled). Run:

        ::

            python label_PVM_pickle.py example.pkl

        A gui should show up, presenting the first frame of the movie. Mouse cursor should
        be surrounded by a box. By left clicking in a given position, the target bounding box is
        saved and next frame is displayed. The size of the current box can be defined by clicking in the
        left top corner of the target object with the middle button, and extending the box to cover the
        entire object. Additionally the box can be scaled up or down using 'a' and 'z' keys. Use right click
        to indicate that the target object is not present in the scene, whenever this is the case. Be careful
        on frames in which target disappears and reappears. If some parts of the movie were not labeled
        correctly, you can move back using cursor keys or scroll bar at the top of the window.

        After the process is complete, press 'q' key and 'y' to save the changes in the file. example.pkl is now
        modified with default target label on the default channel.

    3. Once one channel has been labeled you may want to create e.g. a downscaled version of that channel
        but inherit and scale all the labels. scale_PVM_pickle.py script can be useful for that. Simply run:

        ::

            python scale_PVM_pickle.py example.pkl -i default -o med_res -d 0.5 -s

        Refer to the scale_PVM_pickle.py documentation for details. The options in the example above are:

            * -i input channel - the name of the channel being processed
            * -o output channel -  the name of the new channel that will be created
            * -d 0.5 - scale content by a factor of 0.5
            * -s - set the new channel as default. If the input channel was default and had no other name
                  it will be renamed to channel01.

    4. The file is ready and can be used in tracker benchmark. Refer to TSartifacts documentation
        for information on uploading the file to S3 bucket.

    5. If you want to quickly view the content of the file and labels, use play_PVM_pickle.py script
        simply typing a command:

        ::

            python play_PVM_pickle.py example.pkl -b

        -b option will display the boxes.

Multiple targets
########

The labeled movie format allows to store multiple target labels per channel. This allows to track multiple
objects. In order to label multiple objects follow the steps above, after step 2 execute:

::

    python label_PVM_pickle.py example.pkl -t target1

with the -t option to select the new target. The labeling tool will open without anything labeled. Proceed
labeling the additional objects of interest. Once done, remember to save your work. Files with multiple
labels can also by scaled by the scale_PVM_pickle.py tool. It will scale all the labels obtained from the source
channel.

.. note::

    It's a good habit to carefully label the high resolution version of the video and then use such high
    quality data, to create lower resolution channels. That way the labels in the lower resolution channels
    will be very precise.

If you don't want to modify the source file but instead write to another file, use the -o option when calling the
tool:

::

    python label_PVM_pickle.py example.pkl -t target1 -o new_file.pkl

.. note::

    The labels in the labeled movies can always be modified and improved. Try not to generate too many files
    with the same content but different versions of labels. Instead rather keep everything in one place (file)
    and make sure the labels are of highest possible quality.


Script usage message
########

::

    Usage:
    python label_movie.py input_file.pkl [-o output_file] [-c channel]

    Keys:

       left, right cursor - move to previous/next frame
       t - toggle a tracker
       [space] - validate the tracker bounding box and move to the next frame
       w - write movie to the output file (same as input if not given)
       a - scale current box up by 10% in both dimensions
       z - scale current box down by 10% in both dimensions
       s - scale current box up by 10% in the vertical direction
       x - scale current box down by 10% in the vertical direction
       q - quit


"""
import argparse
from PVM_tools.labeled_movie import FrameCollection
from PVM_tools.bounding_region import BoundingRegion
import cv2
import threading
import logging
import numpy as np
import time


class LabelingApp(object):
    def __init__(self, filename, output=None, channel="default", target="default"):
        self.input_filename = filename
        if output is None:
            self.output_file = self.input_filename
        else:
            self.output_file = output
        self.channel = channel
        self.target = target
        self.reset()

    def reset(self, reload=False):
        cv2.destroyAllWindows()
        fc = FrameCollection()
        if reload:
            fc.load_from_file(filename=self.output_file)
        else:
            fc.load_from_file(filename=self.input_filename)
        self.movie = fc
        self.tracker = None
        self.current_frame_index = 0
        self.right_button_pressed = False
        self.left_button_pressed = False
        self._anchor = None
        self.x_size = 30
        self.y_size = 30
        self._tracked_bounds = None
        self._stored_bounds = None
        self._current_bounds = BoundingRegion()
        # Make sure self.timer is set
        self.schedule_callback()
        # Initialization of the Graphic User Interface
        self.ready_to_refresh = threading.Lock()
        self.timer_lock = threading.Lock()
        self.win_name = 'Labeling video GUI'
        self.image = self.movie.Frame(self.current_frame_index).get_image(channel=self.channel)
        self.display_image = self.movie.Frame(self.current_frame_index).get_image(channel=self.channel)
        self.image_shape = self.image.shape
        self.create_windows()
        self.set_target_absent()
        self.image_buffer = {}
        self._last_update = time.time()
        self.refresh_timer = threading.Timer(0.05, self.refresh)
        self.refresh_timer.start()
        self.needs_refresh = True
        self.trim_end = len(self.movie)-1
        self.trim_start = 0

    def fill_up_the_buffer(self):
        r0 = max(self.current_frame_index-5, 0)
        r1 = min(self.current_frame_index+5, len(self.movie))
        for i in xrange(r0, r1, 1):
            if i not in self.image_buffer.keys():
                self.image_buffer[i] = self.movie.Frame(i).get_image(channel=self.channel)
        for i in self.image_buffer.keys():
            if i < r0 or i > r1:
                del self.image_buffer[i]

    def create_windows(self):
        ''' Create playback progress bar (allow user to move along the movie) '''
        cv2.namedWindow(self.win_name)
        cv2.moveWindow(self.win_name, 150, 150)
        cv2.namedWindow("completeness")
        cv2.moveWindow("completeness", 150, 50)
        cv2.setMouseCallback(self.win_name, self.trackbar_window_onMouse)
        cv2.createTrackbar("Frame", self.win_name, 0, len(self.movie) - 1, self.jump_to_frame_callback)
        cv2.createTrackbar("Trim start", self.win_name, 0, len(self.movie) - 1, self.set_trim_start)
        cv2.createTrackbar("Trim end", self.win_name, len(self.movie) - 1, len(self.movie) - 1, self.set_trim_end)
        im_height = 30  # 30 pixels high completeness bar
        im_width = len(self.movie)
        self.completeness_image = np.zeros((im_height, im_width, 3))
        current_frame_index = int(im_width * float(self.current_frame_index) / len(self.movie))
        self.completeness_image[:, current_frame_index, :] = 1.
        self.completeness = np.zeros(im_width, dtype=np.uint8)
        for i in xrange(len(self.movie)):
            bounds = self.movie.Frame(i).get_label(channel=self.channel, target=self.target)
            if bounds is None or bounds.empty:
                if bounds is None:
                    self.movie.Frame(i).set_label(channel=self.channel, target=self.target, label=BoundingRegion())
                self.completeness[i] = 2
            else:
                if bounds.is_keyframe():
                    self.completeness[i] = 1
                else:
                    self.completeness[i] = 0

    def set_trim_start(self, frame_index):
        self.trim_start = frame_index + 0

    def set_trim_end(self, frame_index):
        self.trim_end = frame_index + 0

    def jump_to_frame_callback(self, frame_index):
        self.current_frame_index = frame_index
        self.image = self.movie.Frame(self.current_frame_index).get_image(channel=self.channel)
        self.needs_refresh = True

    def update_completeness_window(self):
        if self.ready_to_refresh.acquire(False):
            self.completeness_image *= 0
            self.completeness_image[:, self.current_frame_index, :] = 255
            for i in xrange(len(self.movie)):
                    self.completeness_image[:, i, self.completeness[i]] = 255
            cv2.putText(self.completeness_image, '%i' % self.current_frame_index,
                        (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
            cv2.imshow('completeness', self.completeness_image)
            self.ready_to_refresh.release()

    def update_trackbar_window(self):
        if self.ready_to_refresh.acquire():
            image = self.image.copy()
            self._stored_bounds = self.movie.Frame(self.current_frame_index).get_label(channel=self.channel, target=self.target)
            # display rectangle over selected area
            self._current_bounds.draw_box(image, color=(255, 255, 255))
            self._current_bounds.draw_box(image, color=(0, 0, 0), thickness=1)
            # display rectangle over stored area
            if self._stored_bounds is not None:
                self._stored_bounds.draw_box(image, color=(255, 0, 0))
            if self._tracked_bounds is not None:
                self._tracked_bounds.draw_box(image, color=(0, 255, 0))
            # reshape to display if too small
            min_width = 320
            if image.shape[1] < min_width:
                resize_height = int(0.5 + image.shape[0] * min_width / float(image.shape[1]))
                image = cv2.resize(image, (min_width, resize_height), interpolation=cv2.INTER_NEAREST)
            self.display_image=image
            self.ready_to_refresh.release()

    def refresh(self):
        if self.timer_lock.acquire():
            if self.needs_refresh:
                self.update_trackbar_window()
                self.update_completeness_window()
                self.needs_refresh = False
            self.refresh_timer = threading.Timer(0.05, self.refresh)
            self.refresh_timer.start()
            self.timer_lock.release()

    def trackbar_window_onMouse(self, event, x, y, flags, _):
        cv2.imshow(self.win_name, self.display_image)
        # Update state
        self.right_button_pressed = (flags & cv2.EVENT_FLAG_RBUTTON) != 0 and event != cv2.EVENT_RBUTTONUP
        self.left_button_pressed = (flags & cv2.EVENT_FLAG_LBUTTON) != 0 and event != cv2.EVENT_LBUTTONUP

        # Set bounding box
        if event == cv2.EVENT_MBUTTONDOWN:
            if self._anchor is None:
                # Setting the anchor
                self._anchor = (x, y)
            else:
                # Defining the box
                self.save_and_advance()

        elif self._anchor is not None and event != cv2.EVENT_MBUTTONUP:
            # We are creating new bounding box from anchor to current position
            self.x_size = max(5, int(abs(x - self._anchor[0])+0.5))
            self.y_size = max(5, int(abs(y - self._anchor[1])+0.5))
            center_x = (x + self._anchor[0]) / 2
            center_y = (y + self._anchor[1]) / 2
            self.set_bounding_box_with_center(center_x, center_y)
        elif event == cv2.EVENT_MBUTTONUP:
            self.x_size = max(5, int(abs(x - self._anchor[0])+0.5))
            self.y_size = max(5, int(abs(y - self._anchor[1])+0.5))
            center_x = (x + self._anchor[0]) / 2
            center_y = (y + self._anchor[1]) / 2
            self.set_bounding_box_with_center(center_x, center_y)
            keyframe = False
            if flags & cv2.EVENT_FLAG_CTRLKEY != 0:
                keyframe = True
            self.save_and_advance(keyframe=keyframe)
        else:
            self.set_bounding_box_with_center(x, y)

        # If this is a left/right button down event,
        # we can advance one frame immediately
        # and then start the timer
        if event == cv2.EVENT_RBUTTONDOWN:
            self.set_target_absent()
            self.save_and_advance()
            self.right_button_pressed=True
            if self.timer.finished:
                # Rearm the timer
                self.schedule_callback()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.set_bounding_box_with_center(x, y)
            keyframe = False
            if flags & cv2.EVENT_FLAG_CTRLKEY != 0:
                keyframe = True
            self.save_and_advance(keyframe=keyframe)
            self.left_button_pressed=True
            if self.timer.finished:
                # Rearm the timer
                self.schedule_callback()
        elif not (self.right_button_pressed or self.left_button_pressed):
            # No mouse button pressed, so cancel the timer if it's running
            self.timer.cancel()
        if time.time()-self._last_update > 0.1:
            self._last_update = time.time()
            self.needs_refresh = True

    def set_target_absent(self):
        self._current_bounds = BoundingRegion()

    def set_bounding_box_with_center(self, x, y):
        box = [np.clip(int(x+0.5)-self.x_size/2, 0, self.image_shape[1]),
               np.clip(int(y+0.5)-self.y_size/2, 0, self.image_shape[0]),
               self.x_size + min(int(x+0.5)-self.x_size/2, 0),
               self.y_size + min(int(y+0.5)-self.y_size/2, 0)]
        self._current_bounds = BoundingRegion(box=box)
        self.needs_refresh = True

    def schedule_callback(self):
        # 0.2 secs => 5 Hz
        self.timer = threading.Timer(0.1, self.timer_callback)
        self.timer.start()

    def timer_callback(self):
        if self.right_button_pressed or self.left_button_pressed:
            self.save_and_advance()
            self.schedule_callback()

    def advance_current_frame(self, increment=1):
        self.current_frame_index = min(len(self.movie)-1, max(0, self.current_frame_index + increment))
        self.image = self.movie.Frame(self.current_frame_index).get_image(channel=self.channel)
        cv2.setTrackbarPos("Frame", self.win_name, self.current_frame_index)
        if self.tracker is not None:
            self._tracked_bounds = self.tracker.track(self.movie.Frame(self.current_frame_index).get_image(channel=self.channel))

    def save_and_advance(self, keyframe=False):
        self._anchor = None
        self.movie.Frame(self.current_frame_index).set_label(self._current_bounds.copy(), channel=self.channel, target=self.target)
        if self._current_bounds.empty:
            self.completeness[self.current_frame_index] = 2
        else:
            if self._current_bounds.is_keyframe():
                self.completeness[self.current_frame_index] = 1
            else:
                self.completeness[self.current_frame_index] = 0
        if keyframe:
            self.make_keyframe()
            self.advance_current_frame(15)
        else:
            self.advance_current_frame(1)

    def toggle_tracker(self):
        if self.tracker is None:
            self.tracker = CMTVisionTracker()
            self.tracker.prime(self.movie.Frame(self.current_frame_index).get_image(channel=self.channel), self._current_bounds)
            logging.warning("Tracker is enabled")
        else:
            self.tracker = None
            self._tracked_bounds = None
            logging.warning("Tracker is disabled")

    def advance_tracker(self):
        if self._tracked_bounds is not None:
            self._current_bounds=self._tracked_bounds
            self.save_and_advance()

    def export_movie(self):
        trim_frames_front = self.trim_start
        trim_frames_end = (len(self.movie) -1) - self.trim_end
        if trim_frames_end > 0:
            for i in range(trim_frames_end):
                self.movie.delete(-1)
        if trim_frames_front>0:
            for i in range(trim_frames_front):
                self.movie.delete(0)
        logging.warning("Exporting result in file " + self.output_file)
        self.movie.write_to_file(self.output_file)
        logging.warning("Exporting completed !")

    def interpolate(self, start, end):
        print "Interpolating %d %d" % (start, end)
        label0 = self.movie.Frame(start).get_label(channel=self.channel, target=self.target)
        label1 = self.movie.Frame(end).get_label(channel=self.channel, target=self.target)
        if (label0.empty) or (label1.empty):
            return
        box0 = label0.get_box_pixels()
        box1 = label1.get_box_pixels()

        for i in xrange(start+1, end, 1):
            alpha = (end-i)*1.0/(end-start)
            box2 = alpha*np.array(box0)+(1-alpha)*np.array(box1)
            box2 = map(lambda x: int(x), box2)
            self.movie.Frame(i).set_label(BoundingRegion(box=box2, image_shape=self.movie.Frame(i).get_image(channel=self.channel).shape), channel=self.channel, target=self.target)
            self.completeness[i] = 0

    def make_keyframe(self):
        self.completeness[self.current_frame_index] = 1
        self.movie.Frame(self.current_frame_index).get_label(channel=self.channel, target=self.target).set_keyframe(True)

        # find previous keyframe
        next_keyframe = -1
        for i in xrange(self.current_frame_index+1, len(self.movie), 1):
            bounds = self.movie.Frame(i).get_label(channel=self.channel, target=self.target)
            if bounds.is_keyframe():
                next_keyframe = i
                break
        if next_keyframe >= 0:
            print next_keyframe
            self.interpolate(self.current_frame_index, next_keyframe)
        # find next keyframe
        prev_keyframe = -1
        for i in xrange(self.current_frame_index-1, -1, -1):
            bounds = self.movie.Frame(i).get_label(channel=self.channel, target=self.target)
            if bounds.is_keyframe():
                prev_keyframe = i
                break
        if prev_keyframe >= 0:
            print prev_keyframe
            self.interpolate(prev_keyframe, self.current_frame_index)

    def run(self):
        while True:
            key = cv2.waitKey(0)
            cv2.imshow(self.win_name, self.display_image)

            if key == 65361:  # left arrow
                # Go to previous frame
                self.advance_current_frame(-1)
            elif key == 65362:
                self.advance_current_frame(-10)
            elif key == 65363:  # right arrow
                # Go to next frame
                self.advance_current_frame()
            elif key == 65364:
                self.advance_current_frame(10)
            elif key == ord('w'):  # Write
                self.export_movie()
                if self.timer_lock.acquire():
                    if not self.refresh_timer.finished:
                        self.refresh_timer.cancel()
                    self.timer_lock.release()
                self.timer.cancel()
                self.reset()
            elif key == ord('t'):  # Tracker
                self.toggle_tracker()
            elif key == ord(' '):  # Advance tracker
                self.advance_tracker()
                self.needs_refresh = True
            elif key == ord('a'):
                self.x_size = int(self.x_size*1.1)
                self.y_size = int(self.y_size*1.1)
                self.needs_refresh = True
            elif key == ord('z'):
                self.x_size = int(self.x_size*1/1.1)
                self.y_size = int(self.y_size*1/1.1)
                self.needs_refresh = True
            elif key == ord('s'):
                self.y_size = int(self.y_size*1.1)
                self.needs_refresh = True
            elif key == ord('k'):
                self.make_keyframe()
            elif key == ord('x'):
                self.y_size = int(self.y_size*1/1.1)
                self.needs_refresh = True
            elif key == ord('q'):  # Export
                while key != -1:
                    for k in range(10):
                        key = cv2.waitKey(10)
                print "Quiting! Do you want to save your work? [Y/N]"
                while True:
                    key = cv2.waitKey(0)
                    if key == ord('Y') or key == ord('y'):
                        self.export_movie()
                        break
                    if key == ord('N') or key == ord('n'):
                        break
                break
        logging.warning('Exiting')
        self.refresh_timer.cancel()

if __name__ == "__main__":
    doc = """
    Usage:
    python label_movie.py input_file.pkl [-o output_file] [-c channel]

    Keys:

       left, right cursor - move to previous/next frame
       t - toggle a tracker
       [space] - validate the tracker bounding box and move to the next frame
       w - write movie to the output file (same as input if not given)
       a - scale current box up by 10% in both dimensions
       z - scale current box down by 10% in both dimensions
       s - scale current box up by 10% in the vertical direction
       x - scale current box down by 10% in the vertical direction

       q - quit

    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument('input_file',
                        default='demo.pkl',
                        nargs=1,
                        help='Input file')
    parser.add_argument("-o", "--output", type=str, help="Output file (otherwise will save changes in the input file)")
    parser.add_argument("-c", "--channel", type=str, default="default", help="Channel")
    parser.add_argument("-t", "--target", type=str, default="default", help="Name of the labeled target")
    args = parser.parse_args()
    if not args.input_file:
        parser.print_help()
    else:
        print doc
        app = LabelingApp(filename=args.input_file[0],
                          output=args.output,
                          channel=args.channel,
                          target=args.target)
        app.run()
