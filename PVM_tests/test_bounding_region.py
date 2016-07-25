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
import PVM_tools.bounding_region as bounding_region
import numpy as np
import cv2


def test_initialization():
    """
    Test if the bounding regions initialized with a box and a corresponding contour
    are exactly the same in every possible aspect
    """
    shape = (100, 100, 3)
    b = bounding_region.BoundingRegion(shape, box=np.array([10, 20, 15, 45]))
    c = bounding_region.BoundingRegion(shape, contour=np.array([[[10, 20]], [[25, 20]], [[25, 65]], [[10, 65]]]))
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


def test_fundamentals():
    """
    Tests basic funtionality to identify if the behavior of the class
    corresponds to the expected given the inializing data.
    """
    shape = (100, 100, 3)
    image = np.zeros(shape, dtype=np.uint8)
    # Initilized with a box
    b = bounding_region.BoundingRegion(shape, box=np.array([10, 20, 15, 45]))
    b.draw_box(image, (255, 255, 0))
    assert(np.allclose(image[20, 10], np.array([255, 255, 0])))
    assert(np.allclose(image[20, 25], np.array([255, 255, 0])))
    assert(np.allclose(image[65, 25], np.array([255, 255, 0])))
    assert(np.allclose(image[65, 10], np.array([255, 255, 0])))
    # Initilized with a contour
    image = np.zeros(shape, dtype=np.uint8)
    b = bounding_region.BoundingRegion(shape, contour=np.array([[[10, 20]], [[25, 20]], [[25, 65]], [[10, 65]]]))
    b.draw_box(image, (255, 255, 0))
    assert(np.allclose(image[20, 10], np.array([255, 255, 0])))
    assert(np.allclose(image[20, 25], np.array([255, 255, 0])))
    assert(np.allclose(image[65, 25], np.array([255, 255, 0])))
    assert(np.allclose(image[65, 10], np.array([255, 255, 0])))


def test_area():
    """
    Tests if the contour being drawn on the picture has the same area as the one
    used to initialize the class.
    """
    shape = (100, 100, 3)
    image = np.zeros(shape, dtype=np.uint8)
    # Initilized with a box
    b = bounding_region.BoundingRegion(shape, box=np.array([10, 20, 15, 45]))
    b.draw_box(image, (255, 255, 255), thickness=-1)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(imgray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    assert(len(contours) == 1)
    area = cv2.contourArea(contours[0])
    assert(area == b.get_area_pixels())

    # Initilized with a contour
    image = np.zeros(shape, dtype=np.uint8)
    b = bounding_region.BoundingRegion(shape, contour=np.array([[[10, 20]], [[25, 20]], [[25, 65]], [[10, 65]]]))
    b.draw_box(image, (255, 255, 255), thickness=-1)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(imgray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    assert(len(contours) == 1)
    area = cv2.contourArea(contours[0])
    assert(area == b.get_area_pixels())


def test_scaling():
    """
    Tests if the scaling functionality result in correct behavior
    """
    shape = (100, 100, 3)
    # Initilized with a box
    image = np.zeros(shape)
    b = bounding_region.BoundingRegion(shape, box=np.array([40, 40, 20, 20]))
    b.scale(0.5)
    b.draw_box(image, (255, 255, 0))
    assert(np.allclose(image[45, 45], np.array([255, 255, 0])))
    assert(np.allclose(image[45, 55], np.array([255, 255, 0])))
    assert(np.allclose(image[55, 55], np.array([255, 255, 0])))
    assert(np.allclose(image[55, 45], np.array([255, 255, 0])))
    b.scale(2.0)
    b.draw_box(image, (0, 255, 0))
    assert(np.allclose(image[40, 40], np.array([0, 255, 0])))
    assert(np.allclose(image[40, 60], np.array([0, 255, 0])))
    assert(np.allclose(image[60, 40], np.array([0, 255, 0])))
    assert(np.allclose(image[60, 60], np.array([0, 255, 0])))
    b.scale(2.0)
    b.draw_box(image, (0, 255, 255))
    assert(np.allclose(image[30, 30], np.array([0, 255, 255])))
    assert(np.allclose(image[30, 70], np.array([0, 255, 255])))
    assert(np.allclose(image[70, 30], np.array([0, 255, 255])))
    assert(np.allclose(image[70, 70], np.array([0, 255, 255])))
    image = np.zeros(shape)
    # Initilized with a contour
    b = bounding_region.BoundingRegion(shape, contour=np.array([[[40, 40]], [[40, 60]], [[60, 60]], [[60, 40]]]))
    b.scale(0.5)
    b.draw_box(image, (255, 255, 0))
    assert(np.allclose(image[45, 45], np.array([255, 255, 0])))
    assert(np.allclose(image[45, 55], np.array([255, 255, 0])))
    assert(np.allclose(image[55, 55], np.array([255, 255, 0])))
    assert(np.allclose(image[55, 45], np.array([255, 255, 0])))
    b.scale(2.0)
    b.draw_box(image, (0, 255, 0))
    assert(np.allclose(image[40, 40], np.array([0, 255, 0])))
    assert(np.allclose(image[40, 60], np.array([0, 255, 0])))
    assert(np.allclose(image[60, 40], np.array([0, 255, 0])))
    assert(np.allclose(image[60, 60], np.array([0, 255, 0])))
    b.scale(2.0)
    b.draw_box(image, (0, 255, 255))
    assert(np.allclose(image[30, 30], np.array([0, 255, 255])))
    assert(np.allclose(image[30, 70], np.array([0, 255, 255])))
    assert(np.allclose(image[70, 30], np.array([0, 255, 255])))
    assert(np.allclose(image[70, 70], np.array([0, 255, 255])))


def test_box_intersection():
    """
    Simple static test to verify the intersection is indeed an intersection
    """
    shape = (100, 100, 3)
    image = np.zeros(shape, dtype=np.uint8)
    # Initilized with a box
    b1 = bounding_region.BoundingRegion(image_shape=shape, box=np.array([10, 20, 15, 45]))
    b2 = bounding_region.BoundingRegion(image_shape=shape, box=np.array([20, 60, 15, 25]))

    mask1 = b1.get_mask()
    mask2 = b2.get_mask()

    b3 = b1.get_box_intersection(b2)
    b3.set_image_shape(shape=shape)

    mask3 = b3.get_mask()
    mask4 = cv2.bitwise_and(mask1, mask2)
    assert(np.allclose(mask3, mask4))


def test_box_intersection_randomized():
    """
    Deeper test runs 1000 times to verify that the intersection is right each time.
    """
    for i in xrange(1000):
        shape = (100, 100, 3)
        # Initilized with a box
        box1 = np.random.randint(low=10, high=30, size=(4, 1))
        box2 = np.random.randint(low=10, high=30, size=(4, 1))
        b1 = bounding_region.BoundingRegion(image_shape=shape, box=box1)
        b2 = bounding_region.BoundingRegion(image_shape=shape, box=box2)

        mask1 = b1.get_mask()
        mask2 = b2.get_mask()

        b3 = b1.get_box_intersection(b2)
        b3.set_image_shape(shape=shape)

        mask3 = b3.get_mask()
        mask4 = cv2.bitwise_and(mask1, mask2)
        assert(np.allclose(mask3, mask4))


def test_mask_randomized():
    """
    Tests that the mask generation from the box work right.
    """
    for i in xrange(1000):
        shape = (100, 100)
        image = np.zeros(shape, dtype=np.uint8)
        # Initilized with a box
        box1 = np.random.randint(low=10, high=30, size=(4, 1))
        cv2.rectangle(image, (box1[0], box1[1]), (box1[0]+box1[2], box1[1]+box1[3]), color=255, thickness=-1)
        b1 = bounding_region.BoundingRegion(image_shape=shape, box=box1)
        mask1 = b1.get_mask()
        assert (np.allclose(image, mask1))


if __name__ == "__main__":
    test_initialization()
    test_fundamentals()
    test_area()
    test_scaling()
