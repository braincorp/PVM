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
import PVM_tools.labeled_movie as labeled_movie
import numpy as np


def test_basic_channels_1():
    img_left = np.zeros((100, 100, 3))
    img_right = np.zeros((100, 100, 3))+255
    F = labeled_movie.LabeledMovieFrame(internal_storage_method="jpg")
    F.create_channel("left_image")
    F.set_default_channel("left_image")
    F.set_image(img_left)  # Now left_image is the default channel
    F.create_channel("right_image")
    F.set_image(img_right, channel="right_image")
    assert(np.allclose(F.get_image(), img_left))
    assert(np.allclose(F.get_image(channel="left_image"), img_left))
    assert(np.allclose(F.get_image(channel="right_image"), img_right))


def test_basic_channels_2():
    img_left = np.zeros((100, 100, 3))
    img_right = np.zeros((100, 100, 3))+255
    F = labeled_movie.LabeledMovieFrame(internal_storage_method="jpg")
    F.create_channel("left_image")
    F.set_image(img_left, channel="left_image")
    F.create_channel("right_image")
    F.set_image(img_right, channel="right_image")
    F.set_default_channel("left_image")
    assert(np.allclose(F.get_image(), img_left))
    assert(np.allclose(F.get_image(channel="left_image"), img_left))
    assert(np.allclose(F.get_image(channel="right_image"), img_right))


if __name__ == "__main__":
    test_basic_channels_1()
    test_basic_channels_2()
