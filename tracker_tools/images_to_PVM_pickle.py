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
Simple tool to convert sets of images and text ground truth rectangle files into the new format
(somewhat an opposite of the export_to_zip.py file.

Example:
::

    python images_to_PVM_pickle.py -l Example/groundtruth_rect.txt -i Example/img/ -o Example.pkl

Where -l is the path to the file with labels, -i is the directory with images, and -o is the output file.
Additional argument -s may be given to skip certain number of frames (in case the first line in the label file does not
correspond with the first image).

Usage:
::

    usage: images_to_PVM_pickle.py [-h] [-l LABELS] [-i IMAGES] [-o OUTPUT]
                                      [-s SKIP]

    optional arguments:
      -h, --help            show this help message and exit
      -l LABELS, --labels LABELS
                            File containing the rectangles
      -i IMAGES, --images IMAGES
                            Directory containing the images
      -o OUTPUT, --output OUTPUT
                            Output file
      -s SKIP, --skip SKIP  Skip frames (labels only begin at some point)

"""

import argparse
from PVM_tools.bounding_region import BoundingRegion
from PVM_tools.labeled_movie import LabeledMovieFrame, FrameCollection
import os
import cv2
import numpy as np


class ImagesToLabeledMovie(object):

    def __init__(self, label_file=None, image_dir=None, output_file=None, skip=None):
        if label_file is None or image_dir is None or output_file is None:
            raise Exception("Nescessary arguments are missing")
        self.label_file=open(label_file, "r")
        self.image_dir = image_dir
        self.output_file = output_file
        self.frame_collection = FrameCollection()
        self.file_index = 1
        if skip is not None:
            self.skip = int(skip)
        else:
            self.skip = 0

    def get_next_image_file(self):
        p = self.image_dir
        if not p.endswith("/"):
            p += "/"
        p += ("%04d.jpg" % self.file_index)
        self.file_index += 1
        if os.path.isfile(p):
            return p
        else:
            return None

    def run(self):
        i = 0
        while True:
            p = self.get_next_image_file()
            if p is None:
                break
            img = cv2.imread(p)
            if i >= self.skip:
                rect = self.label_file.readline()
                try:
                    box = map(lambda x: int(x), rect.split("\t"))
                    B = BoundingRegion(image_shape=img.shape, box=np.array(box))
                except:
                    try:
                        box = map(lambda x: int(x), rect.split(","))
                        B = BoundingRegion(image_shape=img.shape, box=np.array(box))
                    except:
                        print "No more bounding boxes!"
                        B = BoundingRegion()
            else:
                B = BoundingRegion()
            F = LabeledMovieFrame(internal_storage_method='jpg', compression_level=90)
            F.set_image(img)
            F.set_label(B)
            B.draw_box(img)
            cv2.imshow("image", img)
            cv2.moveWindow("image", 50, 50)
            cv2.waitKey(1)
            i += 1
            self.frame_collection.append(F)
        self.frame_collection.write_to_file(self.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels", type=str, help="File containing the rectangles")
    parser.add_argument("-i", "--images", type=str, help="Directory containing the images")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    parser.add_argument("-s", "--skip", type=str, help="Skip frames (labels only begin at some point)")
    args = parser.parse_args()
    app = ImagesToLabeledMovie(label_file=args.labels,
                               image_dir=args.images,
                               output_file=args.output,
                               skip=args.skip)
    app.run()
