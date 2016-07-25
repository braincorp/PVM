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

import pytest
import logging
import cv2
import numpy as np


def test_tld():
    import other_trackers.tld as tld
    tracker = tld.TLD2()
    initbb = [50, 50, 10, 10]
    img = np.random.randint(0, high=255, size=(100, 100, 3)).astype('uint8')
    height, width = img.shape[:2]
    tracker.set_width_and_height((width, height))
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cvmat = cv2.cv.fromarray(img_grey)
    tracker.selectObject(img_cvmat, tuple(initbb))
    img = np.random.randint(0, high=255, size=(100, 100, 3)).astype('uint8')
    img_cvmat = cv2.cv.fromarray(img)
    tracker.processImage(img_cvmat)
    assert(tracker.getCurrBB() is not None)
    assert(tracker.currConf >= 0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main('-s %s' % __file__)
