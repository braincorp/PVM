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
Convert the data in raw format to a labeled movie.
This script allows for some pre-processing, like rotating or flipping,
changing resolution.

Example:
::

    python raw_to_PVM_pickle.py -i ~/tracker_data/CLIF_2006_SAMPLE_SET/RAW/ -o tmp.pkl -s 2672x4008 -r 90 -d 0.25 -fv -fh -p 000001

This example will take the files in directory ~/tracker_data/CLIF_2006_SAMPLE_SET/RAW/ that begin with 000001,
load them as numpy arrays with shape 2672x4008, rotate by 90 deg, scale by a factor of 0.25, flip vertically, flip
horizontally and save to tmp.pkl.

Usage
::

    usage: raw_to_PVM_pickle.py [-h] [-i IMAGES] [-o OUTPUT] [-c CHANNEL] [-s SHAPE]
                          [-d DSIZE] [-r ROTATE] [-f FORMAT] [-q QUALITY]
                          [-p PREFIX] [-fv] [-fh]

    optional arguments:
      -h, --help            show this help message and exit
      -i IMAGES, --images IMAGES
                            Directory containing the images
      -o OUTPUT, --output OUTPUT
                            Output file
      -c CHANNEL, --channel CHANNEL
                            Output file
      -s SHAPE, --shape SHAPE
                            Shape XxY
      -d DSIZE, --dsize DSIZE
                            Destination size e.g. 500x300 or scale factor e.g 0.5
      -r ROTATE, --rotate ROTATE
                            Rotate by 90, 180 or 270 degrees
      -f FORMAT, --format FORMAT
                            Internal storage format of the pickle
      -q QUALITY, --quality QUALITY
                            Quality 0-100, larger - better image but bigger file
      -p PREFIX, --prefix PREFIX
                            File prefix
      -fv, --flip_vertical  Flip vertical
      -fh, --flip_horizontal
                            Flip horizontal

"""

import argparse
from PVM_tools.labeled_movie import LabeledMovieFrame, FrameCollection
import os
import cv2
import numpy as np


class ImagesToLabeledMovie(object):

    def __init__(self,
                 image_dir=None,
                 output_file=None,
                 skip=None,
                 channel="default",
                 file_prefix="000000",
                 dsize="1.0",
                 flip_vert=None,
                 flip_hor=None,
                 rotate=None,
                 shape=(100, 100),
                 format="jpg",
                 quality=90):

        if image_dir is None or output_file is None:
            raise Exception("Nescessary arguments are missing")
        self.image_dir = image_dir
        self.output_file = output_file
        self.frame_collection = FrameCollection()
        self.channel = channel
        self.file_index = 1
        if skip is not None:
            self.skip = int(skip)
        else:
            self.skip = 0
        self.files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.startswith(file_prefix)])
        self.shape = tuple(map(lambda x: int(x), shape.split("x")))
        self.rotate = rotate
        self.flip_vert = flip_vert
        self.flip_hor = flip_hor
        self.dsize = dsize
        self.format = format
        self.quality = quality

    def get_next_image_file(self):
        if self.file_index < len(self.files):
            self.file_index += 1
            return os.path.join(self.image_dir, self.files[self.file_index-2])
        else:
            return None

    def run(self):
        i = 0
        cv2.namedWindow("image")
        cv2.moveWindow("image", 50, 50)
        while True:
            p = self.get_next_image_file()
            if p is None:
                break
            img = np.memmap(p, dtype=np.uint8, mode="r", shape=self.shape)
            F = LabeledMovieFrame(internal_storage_method=self.format, compression_level=int(self.quality))
            if self.rotate is not None and self.rotate in ["90", "180", "270"]:
                img = np.rot90(img)
                if self.rotate in ["180", "270"]:
                    img = np.rot90(img)
                if self.rotate in ["270"]:
                    img = np.rot90(img)
            if self.flip_vert:
                img = cv2.flip(img, 0)
            if self.flip_hor:
                img = cv2.flip(img, 1)
            if self.dsize.count("x") == 1:
                osize = tuple(map(lambda x: int(x), self.dsize.split("x")))
            else:
                factor = float(self.dsize)
                osize = int(img.shape[1]*factor), int(img.shape[0]*factor)
            img = cv2.resize(img, dsize=osize, interpolation=cv2.INTER_CUBIC)
            F.set_image(img)
            cv2.imshow("image", img)
            cv2.waitKey(40)
            i += 1
            self.frame_collection.append(F)
        self.frame_collection.write_to_file(self.output_file)


if __name__ == "__main__":
    desc = """
    Example:
    python raw_to_PVM_pickle.py -i ~/tracker_data/CLIF_2006_SAMPLE_SET/RAW/ -o tmp.pkl -s 2672x4008 -r 90 -d 0.25 -fv -fh -p 000001
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=str, help="Directory containing the images")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    parser.add_argument("-c", "--channel", type=str, default="default", help="Output file")
    parser.add_argument("-s", "--shape", type=str, default="400x600", help="Shape XxY")
    parser.add_argument("-d", "--dsize", type=str, default="1.0", help="Destination size e.g. 500x300 or scale factor e.g 0.5 ")
    parser.add_argument("-r", "--rotate", type=str, help="Rotate by 90, 180 or 270 degrees ")
    parser.add_argument("-f", "--format", type=str, default="jpg", help="Internal storage format of the pickle")
    parser.add_argument("-q", "--quality", type=str, default="80", help="Quality 0-100, larger - better image but bigger file")
    parser.add_argument("-p", "--prefix", type=str, default="000000", help="File prefix")
    parser.add_argument("-fv", "--flip_vertical", help="Flip vertical", action="store_true")
    parser.add_argument("-fh", "--flip_horizontal", help="Flip horizontal", action="store_true")
    args = parser.parse_args()
    app = ImagesToLabeledMovie(image_dir=args.images,
                               output_file=args.output,
                               channel=args.channel,
                               dsize=args.dsize,
                               rotate=args.rotate,
                               format=args.format,
                               quality=args.quality,
                               flip_vert=args.flip_vertical,
                               flip_hor=args.flip_horizontal,
                               shape=args.shape,
                               file_prefix=args.prefix)
    app.run()
