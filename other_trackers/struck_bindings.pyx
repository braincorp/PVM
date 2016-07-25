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
cimport numpy as np
np.import_array()

cdef extern from "struck.h":
    cdef void struck_init(unsigned char* frame_data, int nrows, int ncols, const BoundingBox& bbox);
    cdef void struck_track(unsigned char* frame_data);
    cdef BoundingBox struck_get_bbox();

    ctypedef struct BoundingBox:
        int width;
        int height;
        int xmin;
        int ymin;

def STRUCK_init(np.ndarray[unsigned char, ndim=3, mode='c'] frame not None, bbox):
    nrows = frame.shape[0]
    ncols = frame.shape[1]
    cdef BoundingBox struck_bbox;
    struck_bbox.xmin = bbox[0]
    struck_bbox.ymin = bbox[1]
    struck_bbox.width = bbox[2]
    struck_bbox.height = bbox[3]
    struck_init(&frame[0, 0, 0], nrows, ncols, struck_bbox);

def STRUCK_track(np.ndarray[unsigned char, ndim=3, mode='c'] frame not None):
    struck_track(&frame[0, 0, 0])

def STRUCK_get_bbox():
    cdef BoundingBox bbox;
    bbox = struck_get_bbox()
    return bbox
