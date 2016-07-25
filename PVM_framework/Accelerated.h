/*# ==================================================================================
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
*/

#include <math.h>

void dot_transpose(double *mult, double* vector, int vect_shape_0, double* matrix, int mat_shape0, int mat_shape1, double* result);
void dot_transpose_simple(double* vector, int vect_shape_0, double* matrix, int mat_shape0, int mat_shape1, double* result);
void derivative_dot(double* vector, double* vector2, int vect_shape_0, double* result);
void derivative_dot_poly(double* vector, double* vector2, int vect_shape_0, double* result);
void generalized_outer(double alpha, double * vector1, int vect1_shape, double * vector2, int vect2_shape, double beta, double* matrix, double* result);
void dot_sigmoid(double* vector, double* matrix, int mat_shape0, int mat_shape1, double beta, double * result, int append_bias);
void dot_sigmoid_poly(double* vector, double* matrix, int mat_shape0, int mat_shape1, double beta, double * result, int append_bias);
void dot_add(double* vector, double* matrix, int mat_shape0, int mat_shape1,  double * result, int append_bias);
void sigmoid_poly(double* result, int shape, double beta);
void sigmoid(double* result, int shape, double beta);