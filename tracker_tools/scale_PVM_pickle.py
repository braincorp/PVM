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
This tool is used to create new channels within a movie containing scaled versions of the original channel.

Channels can either be down or up scaled. All the target labels existing for the original channel are scaled
accordingly.

This can be useful if you have a high definition labeled video but you want to use a downscaled version thereof
for e.g. testing a tracking algorithm.

Usage:
::

    python scale_PVM_pickle.py my_file.pkl -i high_res -o low_res -d 0.5 -s

Arguments:
::

    positional arguments:
      input_file            Input movie file

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT_CHANNEL, --input_channel INPUT_CHANNEL
                            Input channel
      -o OUTPUT_CHANNEL, --output_channel OUTPUT_CHANNEL
                            Output channel
      -d DSIZE, --dsize DSIZE
                            Destination size e.g. 500x300 or 0.5 for downscaling
                            by two
      -f FORMAT, --format FORMAT
                            Internal storage format of the pickle
      -q QUALITY, --quality QUALITY
                            Quality 0-100, larger - better image but bigger file
      -s, --set_default     Set the new channel to be default

"""


import argparse
import cv2
import PVM_tools.labeled_movie as lm


def scale_content(filename, dsize, source_channel, destination_channel, format, grayscale, quality, set_default):
    fc = lm.FrameCollection()
    fc.load_from_file(filename)
    cv2.namedWindow("Image")
    cv2.moveWindow("Image", 50, 50)
    for i in xrange(len(fc)):
        frame = fc.Frame(i)
        im = frame.get_image(channel=source_channel)
        shape = im.shape
        if dsize.count("x") == 1:
            osize = tuple(map(lambda x: int(x), dsize.split("x")))
        else:
            factor = float(dsize)
            osize = int(im.shape[1]*factor), int(im.shape[0]*factor)
        im = cv2.resize(im, dsize=osize, interpolation=cv2.INTER_CUBIC)
        if grayscale:
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im[:, :, 0] = img
            im[:, :, 1] = img
            im[:, :, 2] = img
        frame.create_channel(channel=destination_channel)
        frame.set_image(im, channel=destination_channel, storage_method=format)
        disp_im = im.copy()
        for target in frame.get_targets(channel=source_channel):
            br = frame.get_label(channel=source_channel, target=target)
            br = br.copy()
            br.scale_to_new_image_shape(new_image_shape=im.shape, old_image_shape=shape)
            frame.set_label(br, channel=destination_channel, target=target)
            br.draw_box(disp_im, color=(255, 0, 0), annotation=target)
        if set_default:
            frame.set_default_channel(channel=destination_channel)
        cv2.imshow("Image", disp_im)
        key = cv2.waitKey(1)
        if key == 27:
            quit()
    fc.write_to_file(filename)


if __name__ == "__main__":
    doc = """
    This tool is used to create new channels within a movie containing scaled versions of the original channel.

    Channels can either be down or up scaled. All the target labels existing for the original channel are scaled
    accordingly.

    This can be useful if you have a high definition labeled video but you want to use a downscaled version thereof
    for e.g. testing a tracking algorithm.

    Usage:
    ::

        python scale_PVM_pickle.py my_file.pkl -i high_res -o low_res -d 0.5 -s

    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument('input_file',
                        default='pickle file',
                        nargs=1,
                        help='Input movie file')
    parser.add_argument("-i", "--input_channel", type=str, help="Input channel")
    parser.add_argument("-o", "--output_channel", type=str, default="default", help="Output channel")
    parser.add_argument("-d", "--dsize", type=str, default="1.0", help="Destination size e.g. 500x300 or 0.5 for downscaling by two")
    parser.add_argument("-f", "--format", type=str, default="jpg", help="Internal storage format of the pickle")
    parser.add_argument("-b", "--bw", help="Convert to grayscale", action="store_true")
    parser.add_argument("-q", "--quality", type=str, default="80", help="Quality 0-100, larger - better image but bigger file")
    parser.add_argument("-s", "--set_default", help="Set the new channel to be default", action="store_true")
    args = parser.parse_args()
    if not args.input_file:
        parser.print_help()
    else:
        scale_content(filename=args.input_file[0],
                      dsize=args.dsize,
                      source_channel=args.input_channel,
                      destination_channel=args.output_channel,
                      format=args.format,
                      quality=args.quality,
                      grayscale=args.bw,
                      set_default=args.set_default)
