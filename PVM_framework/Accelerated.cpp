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
# ==================================================================================*/

#include "Accelerated.h"

// Uncomment below for manual loop unrolling
#define UNROLL_LOOP

// Computes mult * vector . matrix.T -> result
void dot_transpose(double* mult, double* vector, int vect_shape_0, double* matrix, int mat_shape0, int mat_shape1, double* result)
{
   for(int i=0; i<mat_shape0; i++)
   {
      int j=0;
      double s0=0;
      int i_mat_shape = i * mat_shape1;
      #ifdef UNROLL_LOOP
      double s1=0;
      double s2=0;
      double s3=0;
      double s4=0;
      double s5=0;
      double s6=0;
      double s7=0;
      for(; j<vect_shape_0-8; j+=8)
      {
         int mat_idx = i_mat_shape+j;
         s0+=vector[j]*matrix[mat_idx];
         s1+=vector[j+1]*matrix[mat_idx+1];
         s2+=vector[j+2]*matrix[mat_idx+2];
         s3+=vector[j+3]*matrix[mat_idx+3];
         s4+=vector[j+4]*matrix[mat_idx+4];
         s5+=vector[j+5]*matrix[mat_idx+5];
         s6+=vector[j+6]*matrix[mat_idx+6];
         s7+=vector[j+7]*matrix[mat_idx+7];
      }
      #endif
      for(; j<vect_shape_0; j++)
      {
         int mat_idx = i_mat_shape+j;
         s0+=vector[j]*matrix[mat_idx];
      }
      #ifdef UNROLL_LOOP
      result[i]=mult[i]*(s0+s1+s2+s3+s4+s5+s6+s7);
      #else
      result[i]=mult[i]*s0;
      #endif
   }
}

// Computes vector . matrix.T -> result
void dot_transpose_simple(double* vector, int vect_shape_0, double* matrix, int mat_shape0, int mat_shape1, double* result)
{
   for(int i=0; i<mat_shape0; i++)
   {
      int j=0;
      double s0=0;
      int i_mat_shape = i * mat_shape1;
      #ifdef UNROLL_LOOP
      double s1=0;
      double s2=0;
      double s3=0;
      double s4=0;
      double s5=0;
      double s6=0;
      double s7=0;
      for(; j<vect_shape_0-8; j+=8)
      {
         int mat_idx = i_mat_shape+j;
         s0+=vector[j]*matrix[mat_idx];
         s1+=vector[j+1]*matrix[mat_idx+1];
         s2+=vector[j+2]*matrix[mat_idx+2];
         s3+=vector[j+3]*matrix[mat_idx+3];
         s4+=vector[j+4]*matrix[mat_idx+4];
         s5+=vector[j+5]*matrix[mat_idx+5];
         s6+=vector[j+6]*matrix[mat_idx+6];
         s7+=vector[j+7]*matrix[mat_idx+7];
      }
      #endif
      for(; j<vect_shape_0; j++)
      {
         int mat_idx = i_mat_shape+j;
         s0+=vector[j]*matrix[mat_idx];
      }
      #ifdef UNROLL_LOOP
      result[i]=(s0+s1+s2+s3+s4+s5+s6+s7);
      #else
      result[i]=s0;
      #endif
   }
}

// Computes vector * (1 - vector) * vector2 -> result
void derivative_dot(double* vector, double* vector2, int vect_shape_0, double* result)
{
   int j=0;
   for(; j<vect_shape_0; j++)
   {
      result[j] = vector[j] * (1 - vector[j]) * vector2[j];
   }
}


void derivative_dot_poly(double* vector, double* vector2, int vect_shape_0, double* result)
{
   int j=0;
   double x;
   for(; j<vect_shape_0; j++)
   {
      if (vector[j]>=0.5) {
         x=(2*vector[j]-1)/(1-(2*vector[j]-1));
      } else {
         x=(2*vector[j]-1)/(1+(2*vector[j]-1));
      }
      x=fabs(x)+1;
      result[j] =(1.0/(2*(x*x))) * vector2[j];
   }
}

// Computes alpha * vector1 x vector2 + beta*matrix -> result
void generalized_outer(double alpha, double * vector1, int vect1_shape, double * vector2, int vect2_shape, double beta, double* matrix, double* result)
{
   for(int i=0; i<vect1_shape; i++)
   {
      int j=0;
      int i_vect = i*vect2_shape;
      #ifdef UNROLL_LOOP
      for(; j<vect2_shape-8; j+=8)
      {
         int mat_ind = i_vect+j;
         result[mat_ind]=alpha*vector1[i]*vector2[j]+beta*matrix[mat_ind];
         result[mat_ind+1]=alpha*vector1[i]*vector2[j+1]+beta*matrix[mat_ind+1];
         result[mat_ind+2]=alpha*vector1[i]*vector2[j+2]+beta*matrix[mat_ind+2];
         result[mat_ind+3]=alpha*vector1[i]*vector2[j+3]+beta*matrix[mat_ind+3];
         result[mat_ind+4]=alpha*vector1[i]*vector2[j+4]+beta*matrix[mat_ind+4];
         result[mat_ind+5]=alpha*vector1[i]*vector2[j+5]+beta*matrix[mat_ind+5];
         result[mat_ind+6]=alpha*vector1[i]*vector2[j+6]+beta*matrix[mat_ind+6];
         result[mat_ind+7]=alpha*vector1[i]*vector2[j+7]+beta*matrix[mat_ind+7];
      }
      #endif
      for(; j<vect2_shape; j++)
      {
         int mat_ind = i_vect+j;
         result[mat_ind]=alpha*vector1[i]*vector2[j]+beta*matrix[mat_ind];
      }
   }
}

void dot_sigmoid(double* vector, double* matrix, int mat_shape0, int mat_shape1, double beta, double * result, int append_bias)
{
   for(int i=0; i<mat_shape1; i++)
   {
      int j=0;
      double s0=0.0;
      #ifdef UNROLL_LOOP
      double s1=0.0;
      double s2=0.0;
      double s3=0.0;
      double s4=0.0;
      double s5=0.0;
      double s6=0.0;
      double s7=0.0;
      for (; j<mat_shape0-8; j+=8)
      {
         int mat_ind = j*mat_shape1+i;
         s0+=vector[j]*matrix[mat_ind];
         s1+=vector[j+1]*matrix[mat_ind+1*mat_shape1];
         s2+=vector[j+2]*matrix[mat_ind+2*mat_shape1];
         s3+=vector[j+3]*matrix[mat_ind+3*mat_shape1];
         s4+=vector[j+4]*matrix[mat_ind+4*mat_shape1];
         s5+=vector[j+5]*matrix[mat_ind+5*mat_shape1];
         s6+=vector[j+6]*matrix[mat_ind+6*mat_shape1];
         s7+=vector[j+7]*matrix[mat_ind+7*mat_shape1];

      }
      #endif
      if (append_bias!=0)
      {
         for (; j<mat_shape0-1; j++)
         {
            int mat_ind = j*mat_shape1+i;
            s0+=vector[j]*matrix[mat_ind];
         }
         s0+=1.0*matrix[(mat_shape0-1)*mat_shape1+i];
      }
      else
      {
         for (; j<mat_shape0; j++)
         {
            int mat_ind = j*mat_shape1+i;
            s0+=vector[j]*matrix[mat_ind];
         }
      }
      #ifdef UNROLL_LOOP
      result[i]=1.0/(1.0+exp(-beta*(s0+s1+s2+s3+s4+s5+s6+s7)));
      #else
      result[i]=1.0/(1.0+exp(-beta*(s0)));
      #endif

   }
}

void dot_sigmoid_poly(double* vector, double* matrix, int mat_shape0, int mat_shape1, double beta, double * result, int append_bias)
{
   for(int i=0; i<mat_shape1; i++)
   {
      int j=0;
      double s0=0.0;
      #ifdef UNROLL_LOOP
      double s1=0.0;
      double s2=0.0;
      double s3=0.0;
      double s4=0.0;
      double s5=0.0;
      double s6=0.0;
      double s7=0.0;
      for (; j<mat_shape0-8; j+=8)
      {
         int mat_ind = j*mat_shape1+i;
         s0+=vector[j]*matrix[mat_ind];
         s1+=vector[j+1]*matrix[mat_ind+1*mat_shape1];
         s2+=vector[j+2]*matrix[mat_ind+2*mat_shape1];
         s3+=vector[j+3]*matrix[mat_ind+3*mat_shape1];
         s4+=vector[j+4]*matrix[mat_ind+4*mat_shape1];
         s5+=vector[j+5]*matrix[mat_ind+5*mat_shape1];
         s6+=vector[j+6]*matrix[mat_ind+6*mat_shape1];
         s7+=vector[j+7]*matrix[mat_ind+7*mat_shape1];

      }
      #endif
      if (append_bias!=0)
      {
         for (; j<mat_shape0-1; j++)
         {
            int mat_ind = j*mat_shape1+i;
            s0+=vector[j]*matrix[mat_ind];
         }
         s0+=1.0*matrix[(mat_shape0-1)*mat_shape1+i];
      }
      else
      {
         for (; j<mat_shape0; j++)
         {
            int mat_ind = j*mat_shape1+i;
            s0+=vector[j]*matrix[mat_ind];
         }
      }
      #ifdef UNROLL_LOOP
      s0=s0+s1+s2+s3+s4+s5+s6+s7;
      #else
      #endif
      result[i]=(s0/(2*(fabs(s0)+1)))+0.5;

   }
}

void dot_add(double* vector, double* matrix, int mat_shape0, int mat_shape1, double * result, int append_bias)
{
   for(int i=0; i<mat_shape1; i++)
   {
      int j=0;
      double s0=0.0;
      #ifdef UNROLL_LOOP
      double s1=0.0;
      double s2=0.0;
      double s3=0.0;
      double s4=0.0;
      double s5=0.0;
      double s6=0.0;
      double s7=0.0;
      for (; j<mat_shape0-8; j+=8)
      {
         int mat_ind = j*mat_shape1+i;
         s0+=vector[j]*matrix[mat_ind];
         s1+=vector[j+1]*matrix[mat_ind+1*mat_shape1];
         s2+=vector[j+2]*matrix[mat_ind+2*mat_shape1];
         s3+=vector[j+3]*matrix[mat_ind+3*mat_shape1];
         s4+=vector[j+4]*matrix[mat_ind+4*mat_shape1];
         s5+=vector[j+5]*matrix[mat_ind+5*mat_shape1];
         s6+=vector[j+6]*matrix[mat_ind+6*mat_shape1];
         s7+=vector[j+7]*matrix[mat_ind+7*mat_shape1];

      }
      #endif
      if (append_bias!=0)
      {
         for (; j<mat_shape0-1; j++)
         {
            int mat_ind = j*mat_shape1+i;
            s0+=vector[j]*matrix[mat_ind];
         }
         s0+=1.0*matrix[(mat_shape0-1)*mat_shape1+i];
      }
      else
      {
         for (; j<mat_shape0; j++)
         {
            int mat_ind = j*mat_shape1+i;
            s0+=vector[j]*matrix[mat_ind];
         }
      }
      #ifdef UNROLL_LOOP
      s0=s0+s1+s2+s3+s4+s5+s6+s7;
      #else
      #endif
      result[i]+=s0;
   }
}

void sigmoid_poly(double* result, int shape, double beta)
{
   for(int i=0; i<shape; i++)
   {
      double s0 = result[i];
      result[i]=(s0/(2*(fabs(s0)+1)))+0.5;
   }
}

void sigmoid(double* result, int shape, double beta)
{
   for(int i=0; i<shape; i++)
   {
      double s0 = result[i];
      result[i]=1.0/(1.0+exp(-beta*(s0)));
   }
}