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
import PVM_framework.PVM_Create as PVM_Create


def test_PVM_crate():
    A0 = PVM_Create.get_fan_in([4, 4], dim_x_l=16, dim_y_l=16, dim_x_u=8, dim_y_u=8, block_x=2, block_y=2)
    assert A0 == [(8, 8), (8, 9), (9, 8), (9, 9)]
    A0 = PVM_Create.get_fan_in([4, 4], dim_x_l=16, dim_y_l=16, dim_x_u=8, dim_y_u=8, block_x=2, block_y=2, radius=1)
    assert A0 == [(8, 8), (8, 9), (9, 8), (9, 9)]
    A0 = PVM_Create.get_fan_in([4, 4], dim_x_l=16, dim_y_l=16, dim_x_u=8, dim_y_u=8, block_x=3, block_y=3)
    assert A0 == [(7, 7), (7, 8), (7, 9), (8, 7), (8, 8), (8, 9), (9, 7), (9, 8), (9, 9)]
    A0 = PVM_Create.get_fan_in([4, 4], dim_x_l=16, dim_y_l=16, dim_x_u=8, dim_y_u=8, block_x=3, block_y=3, radius=1)
    assert A0 == [(7, 8), (8, 7), (8, 8), (8, 9), (9, 8)]
    A0 = PVM_Create.get_fan_in([0, 0], dim_x_l=16, dim_y_l=16, dim_x_u=8, dim_y_u=8, block_x=2, block_y=2)
    assert A0 == [(0, 0), (0, 1), (1, 0), (1, 1)]
    A0 = PVM_Create.get_fan_in([7, 7], dim_x_l=16, dim_y_l=16, dim_x_u=8, dim_y_u=8, block_x=2, block_y=2)
    assert A0 == [(14, 14), (14, 15), (15, 14), (15, 15)]
    A0 = PVM_Create.get_fan_in([0, 0], dim_x_l=20, dim_y_l=20, dim_x_u=8, dim_y_u=8, block_x=2, block_y=2)
    assert A0 == [(0, 0), (0, 1), (1, 0), (1, 1)]
    A0 = PVM_Create.get_fan_in([7, 7], dim_x_l=20, dim_y_l=20, dim_x_u=8, dim_y_u=8, block_x=2, block_y=2)
    assert A0 == [(18, 18), (18, 19), (19, 18), (19, 19)]
    A0 = PVM_Create.get_fan_in([0, 0], dim_x_l=24, dim_y_l=24, dim_x_u=8, dim_y_u=8, block_x=3, block_y=3)
    assert A0 == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    A0 = PVM_Create.get_fan_in([7, 7], dim_x_l=24, dim_y_l=24, dim_x_u=8, dim_y_u=8, block_x=3, block_y=3)
    assert A0 == [(21, 21), (21, 22), (21, 23), (22, 21), (22, 22), (22, 23), (23, 21), (23, 22), (23, 23)]
    A0 = PVM_Create.get_fan_in([0, 0], dim_x_l=20, dim_y_l=20, dim_x_u=8, dim_y_u=8, block_x=3, block_y=3)
    assert A0 == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    A0 = PVM_Create.get_fan_in([7, 7], dim_x_l=20, dim_y_l=20, dim_x_u=8, dim_y_u=8, block_x=3, block_y=3)
    assert A0 == [(17, 17), (17, 18), (17, 19), (18, 17), (18, 18), (18, 19), (19, 17), (19, 18), (19, 19)]
    A0 = PVM_Create.get_fan_in([0, 0], dim_x_l=20, dim_y_l=20, dim_x_u=20, dim_y_u=20, block_x=3, block_y=3)
    assert A0 == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    A0 = PVM_Create.get_fan_in([19, 19], dim_x_l=20, dim_y_l=20, dim_x_u=20, dim_y_u=20, block_x=3, block_y=3)
    assert A0 == [(17, 17), (17, 18), (17, 19), (18, 17), (18, 18), (18, 19), (19, 17), (19, 18), (19, 19)]
    A0 = PVM_Create.get_fan_in([18, 18], dim_x_l=20, dim_y_l=20, dim_x_u=20, dim_y_u=20, block_x=3, block_y=3)
    assert A0 == [(16, 16), (16, 17), (16, 18), (17, 16), (17, 17), (17, 18), (18, 16), (18, 17), (18, 18)]
    A0 = PVM_Create.get_fan_in([0, 0], dim_x_l=5, dim_y_l=5, dim_x_u=1, dim_y_u=1, block_x=3, block_y=3)
    assert A0 == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    A0 = PVM_Create.get_fan_in([0, 0], dim_x_l=2, dim_y_l=2, dim_x_u=1, dim_y_u=1, block_x=2, block_y=2)
    assert A0 == [(0, 0), (0, 1), (1, 0), (1, 1)]
    A0 = PVM_Create.get_fan_in([5, 5], dim_x_l=10, dim_y_l=10, dim_x_u=10, dim_y_u=10, block_x=3, block_y=3, radius=3)
    assert A0 == [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)]
    A0 = PVM_Create.get_fan_in([5, 5], dim_x_l=10, dim_y_l=10, dim_x_u=10, dim_y_u=10, block_x=3, block_y=3, radius=1)
    assert A0 == [(3, 4), (4, 3), (4, 4), (4, 5), (5, 4)]
    A0 = PVM_Create.get_fan_in([5, 5], dim_x_l=10, dim_y_l=10, dim_x_u=10, dim_y_u=10, block_x=2, block_y=2, radius=1)
    assert A0 == [(4, 4), (4, 5), (5, 4), (5, 5)]
    A0 = PVM_Create.get_surround([5, 5], dim_x=10, dim_y=10, radius=1, exclude_self=False)
    assert A0 == [(4, 5), (5, 4), (5, 5), (5, 6), (6, 5)]
    A0 = PVM_Create.get_surround([5, 5], dim_x=10, dim_y=10, radius=1.5, exclude_self=False)
    assert A0 == [(4, 4), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6), (6, 4), (6, 5), (6, 6)]
    A0 = PVM_Create.get_surround([5, 5], dim_x=10, dim_y=10, radius=2, exclude_self=False)
    assert A0 == [(3, 5), (4, 4), (4, 5), (4, 6), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 4), (6, 5), (6, 6), (7, 5)]
    A0 = PVM_Create.get_surround([5, 5], dim_x=10, dim_y=10, radius=1, exclude_self=True)
    assert A0 == [(4, 5), (5, 4), (5, 6), (6, 5)]
    A0 = PVM_Create.get_surround([5, 5], dim_x=10, dim_y=10, radius=1.5, exclude_self=True)
    assert A0 == [(4, 4), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5), (6, 6)]
    A0 = PVM_Create.get_surround([5, 5], dim_x=10, dim_y=10, radius=2, exclude_self=True)
    assert A0 == [(3, 5), (4, 4), (4, 5), (4, 6), (5, 3), (5, 4), (5, 6), (5, 7), (6, 4), (6, 5), (6, 6), (7, 5)]
    A0 = PVM_Create.get_surround([0, 0], dim_x=10, dim_y=10, radius=2, exclude_self=False)
    assert A0 == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]

if __name__ == "__main__":
    test_PVM_crate()
