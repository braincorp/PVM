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
Convert a movie in traditional format (e.g. avi or mov) to a labeled movie archive that can later
be tagged with target labels. This script allows for some pre-processing, like rotating or flipping,
changing resolution.

Example:
::

    python movie_to_pickle -o mymovie.pkl -r 180 -f jpg -q 85 -d 720x460 IMG_1098.MOV

The above will convert a file IMG_1098.MOV, rotate 180 degrees (videos e.g. from iphone will often be upside down)
rescale to 720x460 and save in a pickle as jpg with quality 85.

Usage
::

    usage: movie_to_PVM_pickle.py [-h] [-o OUTPUT] [-c CHANNEL] [-d DSIZE] [-r ROTATE]
                              [-f FORMAT] [-q QUALITY] [-fv] [-fh]
                              input_file

    positional arguments:
      input_file            Input movie file

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            Output file
      -c CHANNEL, --channel CHANNEL
                            Channel
      -d DSIZE, --dsize DSIZE
                            Destination size e.g. 500x300
      -r ROTATE, --rotate ROTATE
                            Rotate by 90, 180 or 270 degrees
      -f FORMAT, --format FORMAT
                            Internal storage format of the pickle
      -q QUALITY, --quality QUALITY
                            Quality 0-100, larger - better image but bigger file
      -fv, --flip_vertical  Flip vertical
      -fh, --flip_horizontal
                            Flip horizontal

"""

import cv2
import PVM_tools.labeled_movie as lm
import argparse
import numpy as np


def convert_to_pickle(infilename, outfilename, dsize, channel, rotate, format, quality, flip_vert, flip_hor):
    cam = cv2.VideoCapture(infilename)
    fc = lm.FrameCollection()
    cv2.namedWindow("Image")
    cv2.moveWindow("Image", 50, 50)
    while True:
        (ret, im) = cam.read()
        if not ret:
            break
        if dsize is not None:
            osize = tuple(map(lambda x: int(x), dsize.split("x")))
            im=cv2.resize(im, dsize=osize, interpolation=cv2.INTER_CUBIC)
        if rotate is not None and rotate in ["90", "180", "270"]:
            im = np.rot90(im)
            if rotate in ["180", "270"]:
                im = np.rot90(im)
            if rotate in ["270"]:
                im = np.rot90(im)
        if flip_vert:
            im = cv2.flip(im, 0)
        if flip_hor:
            im = cv2.flip(im, 1)
        cv2.imshow("Image", im)
        key = cv2.waitKey(1)
        if key == 27:
            quit()
        f = lm.LabeledMovieFrame(internal_storage_method=format, compression_level=int(quality))
        f.create_channel(channel=channel)
        f.set_image(im, channel=channel)
        f.set_default_channel(channel=channel)
        fc.append(f)
    fc.write_to_file(outfilename)


if __name__ == "__main__":
    desc = """
Convert a movie in traditional format (e.g. avi or mov) to a labeled movie archive that can later
be tagged with target labels. This script allows for some pre-processing, like rotating or flipping,
changing resolution.

Example:

python movie_to_pickle -o mymovie.pkl -r 180 -f jpg -q 85 -d 720x460 IMG_1098.MOV

The above will convert a file IMG_1098.MOV, rotate 180 degrees (videos e.g. from iphone will often be upside down)
rescale to 720x460 and save in a pickle as jpg with quality 85.
  """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('input_file',
                        default='some_movie_file',
                        nargs=1,
                        help='Input movie file')
    parser.add_argument("-o", "--output", type=str, help="Output file")
    parser.add_argument("-c", "--channel", type=str, default="default", help="Channel ")
    parser.add_argument("-d", "--dsize", type=str, help="Destination size e.g. 500x300 ")
    parser.add_argument("-r", "--rotate", type=str, help="Rotate by 90, 180 or 270 degrees ")
    parser.add_argument("-f", "--format", type=str, default="jpg", help="Internal storage format of the pickle")
    parser.add_argument("-q", "--quality", type=str, default="80", help="Quality 0-100, larger - better image but bigger file")
    parser.add_argument("-fv", "--flip_vertical", help="Flip vertical", action="store_true")
    parser.add_argument("-fh", "--flip_horizontal", help="Flip horizontal", action="store_true")
    args = parser.parse_args()
    if not args.input_file:
        parser.print_help()
    else:
        convert_to_pickle(infilename=args.input_file[0],
                          outfilename=args.output,
                          channel=args.channel,
                          dsize=args.dsize,
                          rotate=args.rotate,
                          format=args.format,
                          quality=args.quality,
                          flip_vert=args.flip_vertical,
                          flip_hor=args.flip_horizontal)
