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

cdef extern from "xmmintrin.h":
    void _mm_setcsr(unsigned int)
    unsigned int _mm_getcsr()


def set_flush_denormals():
    """
    This call will modify the Control Status Register (CSR) to instruct the CPU to flush denormals.
    A denormal representation of a float is such that is smaller then the smallest actual representation,
    with significand grater or equal to one say e.g.:

    1.0000000xE-50

    With the maximal negative exponent, this looks like the smallest representable number but it is not, since
    one can increase the precission by decreasing the value of significand:

    0.0000001xE-50

    is a few orders of magnitude smaller, but it is not a "normal" floating point represenation. Such denormal
    representations may appear form arithmetic operations, some architectures would flush such small numbers to zero.
    However x86 will by default keep them and perform further arithmetic operations, at the cost of very significant
    slowdown.

    Such slowdown may cause more harm (particularly in a parallel system) than the benefit of increased precision.

    This extension will only build on x86. Flush to zero is default on ARM architecture.
    """
    _mm_setcsr((_mm_getcsr() & ~0x0040) | (0x0040))
