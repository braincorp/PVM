"""
This module contains the class called BoundingRegion which encapsulates
a number of methods used in any bounding box/region related computations. This class is
extensively used in all of the trackers and tracker benchmark.

Import this module as:
::

    import PVM_tools.bounding_region

or:
::

    from PVM_tools.bounding_region import BoundingRegion

"""
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
import cv2
import copy


class BoundingRegion(object):
    """
    A class used to pass return values from trackers as well as to initialize trackers

    :param image_shape: the shape of the image being used
    :param box: initializing box [left, top, width, height] either in pixel or relative coordinates
    :param contour: initializing contour in pixel or relative coordinates

    .. note::
        Contour if given will override the box initialization, so the box will become the bounding box
        of the contour. If the contour is not given, it will be initialized as the set of points of the
        bounding box.

    :Example:
    ::

        import cv2
        import numpy as np

        shape = (100, 100, 3)
        image = np.zeros(shape, dtype=np.uint8)
        d = bounding_region(shape, contour=np.array([[[10, 20]],
                                                     [[25, 15]],
                                                     [[80, 65]],
                                                     [[60, 70]],
                                                     [[20, 75]],
                                                     [[5, 50]]]))
        d.draw_contour(image, color=(0, 255, 0))
        d.draw_box(image)
        min_rect = d.get_min_area_rect_pixels()
        cv2.drawContours(image, [np.int0(cv2.cv.BoxPoints(min_rect))], 0, (0, 255, 255))
        circle = d.get_min_enclosing_circle_pixels()
        cv2.circle(image, circle[0], circle[1], (255, 0, 0))
        cv2.ellipse(image, d.get_ellipse_fit_pixels(), (255, 255, 255))
        cv2.imshow("image", image)
        cv2.waitKey(0)


    A BoundingRegion object can also be initilized with a box (rectangle):
    :Example:
    ::

        shape = (100, 100, 3)
        image = np.zeros(shape, dtype=np.uint8)
        b = BoundingRegion(shape, box=np.array([10, 20, 15, 45]))
        b.draw_box(image)
        cv2.ellipse(image, b.get_ellipse_fit_pixels(), (255, 255, 255))
        cv2.imshow("image", image)
        cv2.waitKey(0)

    When a BoundingRegion object is initiated without parameters, it will assume it is empty. The property
    BoundingRegion.empty will be set to true, and object methods will either be returning None or have no effect.
    :Example:
    ::

        e = BoundingRegion()
        box = e.get_box_pixels()
        # box is None
        e.draw_box(image)
        # nothing happens to the image

    """

    def __init__(self, image_shape=None, box=None, contour=None, confidence=0.0):
        self._version = 0.1
        self._confidence = confidence
        self._is_keyframe = False
        self.image_shape = image_shape
        self.box_is_primary = False
        self.box = None
        self.contour = None
        self._empty = False
        if image_shape is not None:
            self.set_image_shape(image_shape)
        if box is None and contour is None:
            self._empty = True
            return
        if box is not None:
            if type(box) != np.ndarray:
                # if a list of floats are passed in this will fail
                box = np.array(box, dtype=np.int32)

            if type(box) == BoundingRegion:
                box = box.get_box_pixels()

            if np.any(np.isinf(box)) or np.any(np.isnan(box)):
                raise Exception("Bounding Box contains inf or nan")

            if np.min(box[2:]) < 0:
                self._empty = True
                return

            if np.issubdtype(box.dtype, np.float):
                # box is stored in pixel coordinates, for now at least
                assert np.max(np.abs(box)) < 1.5, "relative bounding box values used but are much larger than 1.0"
                
                self.box = (box * self.image_shape_factor).astype(np.int32)
            else:
                self.box = box
            self.box_is_primary = True

        if contour is not None:
            if type(contour) == BoundingRegion:
                contour = contour.get_contour_pixels()

            if np.issubdtype(contour.dtype, np.float):
                contour_int = np.zeros_like(contour, dtype=np.int32)
                for i in contour.shape[0]:
                    contour_int[i][0][0] = int(self.image_shape[1]*contour[i][0][0])
                    contour_int[i][0][1] = int(self.image_shape[0]*contour[i][0][1])
                self.contour = contour_int
            else:
                self.contour = contour
            self._box_from_contour()
        else:
            self._contour_from_box()
        self._update_internals()

    def _contour_from_box(self):
        contour = np.zeros((4, 1, 2), dtype=np.int32)
        contour[0][0][0] = self.box[0]
        contour[0][0][1] = self.box[1]
        contour[1][0][0] = self.box[0]+self.box[2]
        contour[1][0][1] = self.box[1]
        contour[2][0][0] = self.box[0]+self.box[2]
        contour[2][0][1] = self.box[1]+self.box[3]
        contour[3][0][0] = self.box[0]
        contour[3][0][1] = self.box[1]+self.box[3]
        self.contour = contour
        self.box_is_primary = True

    def _box_from_contour(self):
        self.box_is_primary = False
        self.box = np.array(cv2.boundingRect(self.contour)) - np.array([0, 0, 1, 1])

    def _update_internals(self):
        if not self._empty:
            self.moments = cv2.moments(self.contour)

    def set_image_shape(self, shape):
        """
        Sets the image shape. Image shape is nescessary to calculate the relative coordinates.

        :param shape: tuple, e.g. (100, 100, 3)
        :return:
        """
        self.image_shape = shape
        self.image_shape_factor = np.array(list(self.image_shape[1::-1])*2)

    def get_box_pixels(self):
        """
        Returns the bounding box in image pixel coordinates or None if the object is empty

        :return: bounding box [t, l, w, h]
        :rtype: numpy.ndarray of int32
        """
        if self._empty:
            return None
        return (self.box[0], self.box[1], self.box[2], self.box[3])

    def get_box_relative(self):
        """
        Returns the bounding box in relative coordinates or None if the object is empty

        :return: bounding box [t, l, w, h]
        :rtype: numpy.ndarray of float
        """
        if self._empty:
            return None
        if self.image_shape is None:
            raise Exception("Image shape is nescessary to compute the relative coordinates")
        return self.box.astype(np.float) / self.image_shape_factor

    def get_contour_pixels(self):
        """
        Returns the bounding contour in image pixel coordinates or None if the object is empty

        :return: contour, array N x 1 x 2 representing the contour as a collection of N points
        :rtype: numpy.ndarray of int32
        """
        if self._empty:
            return None
        return self.contour

    def get_contour_relative(self):
        """
        Returns the contour in relative coordinates or None if the object is empty

        :return: contour, array N x 1 x 2 representing the contour as a collection of N points
        :rtype: numpy.ndarray of float
        """
        if self._empty:
            return None
        if self.image_shape is None:
            raise Exception("Image shape is nescessary to compute the relative coordinates")
        contour_f = np.zeros_like(self.contour, dtype=np.float)
        for i in xrange(self.contour.shape[0]):
            contour_f[i][0][0] = self.contour[i][0][0]*1.0/self.image_shape[1]
            contour_f[i][0][1] = self.contour[i][0][1]*1.0/self.image_shape[0]
        return contour_f

    def get_box_center_pixels(self):
        """
        Returns the center of the box in pixel coordinates or None if the object is empty

        :return: center tuple (x, y) of int
        :rtype: 2-tuple of integers
        """
        if not self._empty:
            return self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2
        else:
            return None

    def get_box_center_relative(self):
        """
        Returns the center of the box in pixel coordinates or None if the object is empty

        :return: center tuple (x, y) of float
        :rtype: 2-tuple of floats
        """
        if self.image_shape is None:
            raise Exception("Image shape is nescessary to compute the relative coordinates")
        if not self._empty:
            (bx, by) = self.get_box_center_pixels()
            return (bx *1.0 / self.image_shape[1], by * 1.0 / self.image_shape[0])
        else:
            return None

    def get_centroid_pixels(self):
        """
        Returns the centroid of the contour in pixel coordinates or None if the object is empty

        :return: centroid tuple (x, y) of int
        :rtype: 2-tuple of integers
        """
        if self._empty:
            return None
        cx = int(self.moments['m10']/self.moments['m00'])
        cy = int(self.moments['m01']/self.moments['m00'])
        return cx, cy

    def get_centroid_relative(self):
        """
        Returns the centroid of the contour in relative coordinates. Returns None if the object is empty.

        :return: centroid tuple (x, y) of float
        :rtype: 2-tuple of floats
        """
        if self._empty:
            return None
        if self.image_shape is None:
            raise Exception("Image shape is nescessary to compute the relative coordinates")
        (cx, cy) = self.get_centroid_pixels()
        return (cx * 1.0 / self.image_shape[1], cy * 1.0 / self.image_shape[0])

    def scale(self, factor_x, factor_y=None):
        """
        Expand or contract the bounding box or contour around its center by a given factor

        :param factor_x: The multiplicative scale parameter in the x direction
        :type factor_x: float
        :param factor_y: The multiplicative scale parameter in the y direction
        :type factor_y: float

        .. note::
            if factor_y parameter is omitted, then the factor_x is used in both directions

        .. note::
            The scaling is done with respect to the contour's centroid as computed by the get_centroid
            methods.

        :Example:
        ::

            shape = (100, 100, 3)
            image = np.zeros(shape, dtype=np.uint8)
            d = bounding_region(shape, contour=np.array([[[10, 20]],
                                                         [[25, 15]],
                                                         [[80, 65]],
                                                         [[60, 70]],
                                                         [[20, 75]],
                                                         [[5, 50]]]))
            d.draw_contour(image, color=(0, 255, 0))
            # Scale to half the size
            d.scale(0.5)
            d.draw_contour(image, color=(255, 255, 0))
            d.draw_box(image)
            cv2.imshow("Two contours", image)
            cv2.waitKey(0)

        """
        if self._empty:
            return
        if factor_y is None:
            factor_y = factor_x
        if self.image_shape is None:
            raise Exception("Image shape is nescessary to compute the relative coordinates")
        if self.box_is_primary:
            shift_x = self.box[2] * (1.-factor_x) * 0.5
            shift_y = self.box[3] * (1.-factor_y) * 0.5
            self.box = np.array([np.maximum(self.box[0]+shift_x, 0),
                                 np.maximum(self.box[1]+shift_y, 0),
                                 np.minimum(self.box[2]*factor_x, self.image_shape[1]),
                                 np.minimum(self.box[3]*factor_y, self.image_shape[0])]).astype(np.int32)
            self._contour_from_box()
            self._update_internals()
        else:
            (cx, cy) = self.get_centroid_pixels()
            new_contour = np.zeros_like(self.contour, dtype=np.int32)
            for i in xrange(self.contour.shape[0]):
                new_contour[i][0][0] = np.clip(int(cx + (self.contour[i][0][0]-cx)*factor_x), a_min=0, a_max=self.image_shape[1])
                new_contour[i][0][1] = np.clip(int(cy + (self.contour[i][0][1]-cy)*factor_y), a_min=0, a_max=self.image_shape[0])
            self.contour = new_contour
            self._box_from_contour()
            self._update_internals()

    def draw_box(self, image, color=(0, 0, 255), thickness=2, annotation=None, linetype=8):
        """
        Draws the bounding box on an image

        :param image: image to place the drawing (RGB)
        :type image: numpy.ndarray (w, h, 3)
        :param color: 3-tuple representing RGB of the color
        :type color: 3-tuple
        :param thickness: line thickness (default=2)
        :type thickness: int
        :param annotation: optional annotation (e.g. name of the tracker)
        :type annotation: basestring
        :return: No return value

        :Example:
        ::

            shape = (100, 100, 3)
            image = np.zeros(shape, dtype=np.uint8)
            d = bounding_region(shape, contour=np.array([[[10, 20]],
                                                         [[25, 15]],
                                                         [[5, 50]]]))
            d.draw_box(image, color=(0, 255, 0), annotation="This is a box")


        """
        if self._empty:
            return
        cv2.rectangle(image,
                      (self.box[0], self.box[1]),
                      (self.box[0] + self.box[2], self.box[1] + self.box[3]),
                      color=color,
                      thickness=thickness,
                      lineType=linetype)
        if annotation is not None:
            cv2.putText(image, text=annotation, org=(self.box[0], self.box[1]+8), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.7, color=color, lineType=linetype)

    def draw_contour(self, image, color=(0, 0, 255), thickness=2, linetype=8):
        """
        Draws the bounding contour on an image

        :param image: image to place the drawing (RGB)
        :type image: numpy.ndarray (w, h, 3)
        :param color: 3-tuple representing RGB of the color
        :type color: 3-tuple
        :param thickness: line thickness (default=2)
        :type thickness: int
        :return: No return value
        """
        if self._empty:
            return
        cv2.drawContours(image, [self.contour], 0, color=color, thickness=thickness, lineType=linetype)

    def get_min_area_rect_pixels(self):
        """
        Calculates the minimum enclosing area rectangle (can be rotated). Returns None if the object is empty.

        :return: tuple composed of (x, y) of the center, (w, h) and the angle ((x, y), (w, h), angle)
        :rtype: 3-tuple
        """
        if self._empty:
            return None
        rect = cv2.minAreaRect(self.contour)
        return rect

    def get_min_enclosing_circle_pixels(self):
        """
        Calculates the minimum enclosing circle. Returns None if the object is empty.

        :return: tuple composed of (x, y) of the center and the radius ((x, y), radius)
        :rtype: 3-tuple of integer
        """
        if self._empty:
            return None
        circle = cv2.minEnclosingCircle(self.contour)

        return (int(circle[0][0]), int(circle[0][1])), int(circle[1])

    def get_ellipse_fit_pixels(self):
        """
        Calculates the minimum enclosing ellipse
        """
        if self._empty:
            return None
        ellipse = cv2.fitEllipse(self.contour)
        return ellipse

    def get_area_pixels(self):
        """
        Returns the area of the enclosing contour in pixels^2 or None if the object is empty

        :return: area
        :rtype: int
        """
        if self._empty:
            return 0
        area = cv2.contourArea(self.contour)
        return area

    def get_area_relative(self):
        """
        Returns the area of the enclosing contour in fraction of the area of the whole image or None if the object is
        empty

        :return: area
        :rtype: float
        """
        if self._empty:
            return None
        if self.image_shape is None:
            raise Exception("Image shape is nescessary to compute the relative coordinates")
        area = self.get_area_pixels()
        return float(area)/(self.image_shape[0]*self.image_shape[1])

    def get_perimeter_pixels(self):
        """
        Returns the perimeter of the enclosing contour in pixels or None if the object is empty

        :return: perimeter
        :rtype: int
        """
        if self._empty:
            return None
        perimeter = cv2.arcLength(self.contour, closed=True)
        return perimeter

    @property
    def empty(self):
        """
        Returns true if the bounding box is empty. If the object is empty many methods will return None or have no effect.

        :return: True | False
        :rtype: Boolean
        """
        return self._empty

    @property
    def confidence(self):
        """
        Returns the optional confidence that the region is indeed right, if the tracking algorithm did provide it.

        :return: confidence
        :rtype: float
        """
        return self._confidence

    def get_mask(self, contour=False):
        """
        Returns a mask, single channel uint8 numpy array with values 255 inside the bounding box or bounding
        contour and zeros elsewhere.

        :param contour: if true the bounding contour will be painted, otherwise just the bounding box. Default false.
        :type contour: bool
        :return: mask
        :rtype: numpy.ndarray
        """
        if self.image_shape is None:
            raise Exception("Image shape is nescessary to compute the relative coordinates")
        mask = np.zeros((self.image_shape[0], self.image_shape[1]), dtype=np.uint8)
        if self._empty:
            return mask
        if contour:
            cv2.drawContours(mask, [self.contour], 0, color=255, thickness=-1)
        else:
            cv2.rectangle(mask,
                          (self.box[0], self.box[1]),
                          (self.box[0] + self.box[2], self.box[1] + self.box[3]),
                          color=255,
                          thickness=-1)
        return mask

    def get_box_intersection(self, bounding_region):
        """
        Given a bounding_region object computes and returns a new BoundingRegion that
        corresponds to the intersection of the bounding box of the current object with the
        box of the region given as argument. Retuns an empty BoundingRegion if the interseciton
        is empty.

        :param bounding_region: A BoundingRegion object to compute intersection with
        :type bounding_region: BoundingRegion
        :return: Bounding region of the intersection of the boxes
        :rtype: BoundingRegion
        """
        x1_1 = self.box[0]
        y1_1 = self.box[1]
        x1_2 = self.box[0] + self.box[2]
        y1_2 = self.box[1] + self.box[3]
        box2 = bounding_region.get_box_pixels()
        x2_1 = box2[0]
        y2_1 = box2[1]
        x2_2 = box2[0] + box2[2]
        y2_2 = box2[1] + box2[3]

        x3_1 = max(x1_1, x2_1)
        y3_1 = max(y1_1, y2_1)
        width = max(-1, min(x1_2, x2_2) - x3_1)
        height = max(-1, min(y1_2, y2_2) - y3_1)
        if width * height >= 0:
            return BoundingRegion(image_shape=self.image_shape, box=(x3_1, y3_1, width, height))
        else:
            return BoundingRegion()

    def copy(self):
        """
        Returns a copy of the object.

        :return: copy
        :rtype: BoundingRegion
        """
        return copy.deepcopy(self)

    def scale_to_new_image_shape(self, new_image_shape, old_image_shape=None):
        """
        Converts a bounding region to an instance which corresponds to the same relative
        size of the bounding box in a new size image. E.g. a bounding region may correspond
        to a box (100, 100, 50, 50) in an 1080x720 frame. Now the frame is scaled to 540x360, the new bounding
        region should be scaled to (50, 50, 25, 25) to correspond with the same area in the image.

        :param new_image_shape: the new shape of the image
        :type new_image_shape: tuple
        :param old_image_shape: old shape (optional if not given earlier)
        :type old_image_shape: tuple
        """
        if self.image_shape is None and old_image_shape is None:
            raise Exception("Image shape is nescessary to compute the relative coordinates")
        if old_image_shape is None:
            old_image_shape = self.image_shape
        if self._empty:
            return
        factor_x = new_image_shape[1]*1.0/old_image_shape[1]
        factor_y = new_image_shape[0]*1.0/old_image_shape[0]
        self.image_shape = new_image_shape
        self.image_shape_factor = np.array(list(self.image_shape[1::-1])*2)
        if self.box_is_primary:
            self.box = np.array([np.maximum(self.box[0]*factor_x, 0),
                                 np.maximum(self.box[1]*factor_y, 0),
                                 np.minimum(self.box[2]*factor_x, new_image_shape[1]),
                                 np.minimum(self.box[3]*factor_y, new_image_shape[0])]).astype(np.int32)
            self._contour_from_box()
            self._update_internals()
        else:
            new_contour = np.zeros_like(self.contour, dtype=np.int32)
            for i in xrange(self.contour.shape[0]):
                new_contour[i][0][0] = np.clip(int((self.contour[i][0][0])*factor_x), a_min=0, a_max=new_image_shape[1])
                new_contour[i][0][1] = np.clip(int((self.contour[i][0][1])*factor_y), a_min=0, a_max=new_image_shape[0])
            self.contour = new_contour
            self._box_from_contour()
            self._update_internals()

    def get_version(self):
        """
        Gets the version of the bounding region

        :return:
        """
        if hasattr(self, "_version"):
            return self._version
        else:
            return 0

    def is_keyframe(self):
        """
        Is a keyframe

        :return:
        """
        if hasattr(self, "_is_keyframe"):
            return self._is_keyframe
        else:
            return False

    def set_keyframe(self, key_status):
        """
        Set keyframe status

        :param key_status:
        :return:
        """
        self._is_keyframe = key_status


if __name__ == "__main__":
    shape = (100, 100, 3)
    image = np.zeros(shape)
    b = BoundingRegion(shape, box=np.array([10, 20, 15, 45]))
    f = BoundingRegion(shape, box=np.array([10, 20, 15, 45]))
    c = BoundingRegion(shape, contour=np.array([[[10, 20]], [[25, 20]], [[25, 65]], [[10, 65]]]))
    assert(np.allclose(b.get_box_pixels(), c.get_box_pixels()))
    assert(np.allclose(b.get_contour_pixels(), c.get_contour_pixels()))
    assert(np.allclose(b.get_contour_relative(), c.get_contour_relative()))
    assert(b.get_box_center_pixels() == c.get_box_center_pixels())
    assert(b.get_box_center_relative() == c.get_box_center_relative())
    assert(b.get_centroid_pixels() == c.get_centroid_pixels())
    assert(b.get_centroid_relative() == c.get_centroid_relative())
    assert(b.get_area_pixels() == c.get_area_pixels())
    assert(b.get_area_relative() == c.get_area_relative())
    assert(b.get_perimeter_pixels() == c.get_perimeter_pixels())

    print b.get_box_pixels()
    print b.get_contour_pixels()
    print b.get_contour_relative()
    print b.get_box_center_pixels()
    print b.get_box_center_relative()
    print b.get_centroid_pixels()
    print b.get_centroid_relative()
    print b.get_area_pixels()
    print b.get_area_relative()
    print b.get_perimeter_pixels()
    f.scale(0.5)
    b.draw_box(image)
    f.draw_box(image, (255, 255, 0))
    f.scale(2)
    f.draw_box(image, (255, 255, 0))
    cv2.imshow("image", image)
    cv2.waitKey(0)

    image = np.zeros(shape)
    shape = (100, 100, 3)
    d = BoundingRegion(shape, contour=np.array([[[10, 20]], [[25, 15]], [[80, 65]], [[60, 70]], [[20, 75]], [[5, 50]]]))
    d.draw_contour(image, color=(0, 255, 0), thickness=-1)
    d.scale(0.5)
    d.draw_contour(image, color=(255, 255, 0))
    d.draw_box(image)
    cv2.drawContours(image, [np.int0(cv2.cv.BoxPoints(d.get_min_area_rect_pixels()))], 0, (0, 255, 255))
    circle = d.get_min_enclosing_circle_pixels()
    cv2.circle(image, circle[0], circle[1], (255, 0, 0))
    cv2.ellipse(image, d.get_ellipse_fit_pixels(), (255, 255, 255))
    cv2.imshow("image1", image)
    cv2.waitKey(0)
