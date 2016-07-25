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
import time
import cv2


if __name__ == "__main__":
    import other_trackers.struck_bindings as struck
    cam = cv2.VideoCapture(0)
    for i in range(40):
        ret, frame = cam.read()
        time.sleep(0.1)

    height = frame.shape[0]
    width = frame.shape[1]

    bbox_width = 80
    bbox_height = 80

    bbox = [(width-bbox_width)/2, (height-bbox_height)/2, bbox_width, bbox_height]
    struck.STRUCK_init(frame, bbox)

    while (True):
        ret, frame = cam.read()
        struck.STRUCK_track(frame)
        bbox = struck.STRUCK_get_bbox()
        cv2.rectangle(frame, (bbox["xmin"], bbox["ymin"]), (bbox["xmin"]+bbox["width"], bbox["ymin"]+bbox["height"]), (0, 1, 0))
        cv2.imshow('original', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
