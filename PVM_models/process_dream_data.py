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
import PVM_framework.CoreUtils as CoreUtils
import numpy as np
import cv2
import sys
import argparse


def disp(winname, im):
    res = cv2.resize((im*255).astype(np.uint8), dsize=(200, 200))
    cv2.imshow(winname, res)


def normalized_dot(a, b):
    return np.dot(a.flatten(), b.flatten())/(np.sqrt(np.dot(a.flatten(), a.flatten()))*np.sqrt(np.dot(b.flatten(), b.flatten())))


def rotate(li, x):
    return li[-x % len(li):] + li[:-x % len(li)]


def process_data(filename, s):
    """
    Processes the information collected in a dream experiment from a PVM model.

    A sequence of frames obtained in a dream mode is compared with the original seqience of frames.
    All the pairwise correlations are computed and displayed in a 2d plot. A light colored diagonal feature (top-left to bottom
    right diagonal) corresponds with a recreated subsequence of the original sequence.
    :param filename:
    :param s:
    :return:
    """
    dream_data = CoreUtils.load_model(filename)
    frames = len(dream_data["stage0_data"])
    dot_im = np.zeros((frames, frames))
    dot_im_self = np.zeros((frames, frames))
    e = np.exp(1)
    dream_data["stage0_data"] = rotate(dream_data["stage0_data"], -s)
    for x in range(frames):
        for y in range(frames):
            dot_im[x, y] = (np.exp(normalized_dot(dream_data["stage0_data"][x][1], dream_data["stage1_data"][y]))-1.0)/e
        print "*",
        sys.stdout.flush()
    dot_im_int = (255*dot_im).astype(np.uint8)
    cv2.putText(dot_im_int, "Sequence shift %d" % s, (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 0), lineType=cv2.CV_AA)
    # dot_im_int_self = (255*dot_im_self).astype(np.uint8)
    cv2.imwrite("dream_correlation_%05d.jpg" % s, dot_im_int)
    # cv2.imwrite("dream_self_correlation.jpg", dot_im_int)
    cv2.imshow("Dream correlation", dot_im_int)
    # cv2.imshow("Dream self correlation", dot_im_int_self)
    for x in range(10):
        cv2.waitKey(30)
    print "--------------------------"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Movie file to load. If empty the script will try to use the camera", type=str, default="")
    parser.add_argument("-s", "--shift", help="Shift from the beginning of the sequence", type=str, default="0")
    parser.add_argument("-r", "--range", help="Range of the files", type=str, default="")
    args = parser.parse_args()
    if args.range == "":
        process_data(args.file, int(args.shift))
    else:
        for i in range(0, 124):
            s = i*10
