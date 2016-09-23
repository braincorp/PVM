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
This tool is used to play a pickled labeled frame collection

Example:
::

    python play_PVM_pickle.py some_pickle.pkl -b -c some_channel

The -b option will display the bounding boxes of all the labeled targets, -c selects the channel to play.

"""

import argparse
from PVM_tools.labeled_movie import FrameCollection
import cv2
import numpy as np
import os


if __name__ == "__main__":
    desc = """
    This tool is used to play a pickled labeled frame collection
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('input_file',
                        nargs="*",
                        default='demo.pkl',
                        help='Input file')
    parser.add_argument("-b", "--box", help="Draw the bounding box", action="store_true")
    parser.add_argument("-c", "--channel", type=str, default="default", help="Channel ")
    parser.add_argument("-o", "--output", type=str, default="", help="Output file")
    parser.add_argument("-k", "--crop", action="store_true", help="Crop to bounding box, black frame is target absent")
    parser.add_argument("-s", "--size", type=str, default="200x200", help="Size of the crop, default 200x200")
    parser.add_argument("-t", "--target", type=str, default="default", help="Crop to bounding box, black frame is target absent")

    args = parser.parse_args()
    if not args.input_file:
        parser.print_help()
    else:
        cv2.namedWindow("Player")
        cv2.moveWindow("Player", 50, 50)
        box_shape = tuple(map(lambda x: int(x), args.size.split("x")))
        _video = None
        fc = FrameCollection()
        for i in range(len(args.input_file)):
            fc.load_from_file(filename=args.input_file[i])
            fc.set_active_channel(args.channel)
            for i in xrange(len(fc)):
                img = fc.Frame(i).get_image(channel=args.channel)
                if args.box:
                    for target in fc.Frame(i).get_targets():
                        br = fc.Frame(i).get_label(channel=args.channel, target=target)
                        br.draw_box(img, thickness=1, annotation=target)
                if args.crop:
                    br = fc.Frame(i).get_label(channel=args.channel, target=args.target)
                    if not br.empty:
                        box = br.get_box_pixels()
                        crop = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                        crop_res = cv2.resize(crop, dsize=box_shape, interpolation=cv2.INTER_CUBIC)
                        img = crop_res
                    else:
                        img = np.zeros((box_shape[0], box_shape[1], 3), dtype=np.uint8)
                elif args.size:
                    img = cv2.resize(img, dsize=box_shape, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Player", img)
                if args.output != "":
                    if _video is None:
                        _video = cv2.VideoWriter()
                        fps = 20
                        retval = _video.open(os.path.expanduser(args.output),
                                             cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),
                                             fps, (img.shape[1], img.shape[0]))
                        assert(retval)
                    _video.write(img)
                key = cv2.waitKey(int(1000/fc.fps)) & 0xFF
                if key == 27:
                    break
