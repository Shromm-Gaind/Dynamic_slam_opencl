#ifndef KERNELS_H
#define KERNELS_H

// Declarations of local device functions used by the kernels.

// Tau_HSV_grad(I) := distance in 8D space [ sin(Hue ), cos(Hue ), Saturation , (Saturation)/dx , (Saturation)/dy , Value , (Value)/dx , (Vallue)/dy ]
float Tau_HSV_grad (float8 B, float8 c);

float8 Tau_HSV_grad_8chan (float8 B, float8 c);

///////////////////// Interpolation /////////////////////

float8 nearest_neigbour (float8* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);

float8 bilinear (__global float8* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);

float trilinear (__global float* vol, float u_flt, float v_flt, float layer_flt, int mm_pixels, int cols, int read_offset_, uint reduction);

float8 bilinear_SE3_grad (__global float8* img, float u_flt, float v_flt, int cols, int read_offset_); //, , uint reduction, int i, int mm_pixels);

float4 bilinear_flt4 (__global float4* img, float u_flt, float v_flt, int cols, int read_offset_);                                   // Used in tracking

void bilinear_SE3_grad_weight (float4 weights[6], __global float8* SE3_grad_map_cur_frame, int read_index, __global float8* SE3_grad_map_new_frame, float u2_flt, float v2_flt, int cols, int read_offset_, uint reduction, uint mm_pixels, float alpha );

float bilinear_grad_weight (__global float8* HSV_grad, int read_index, float u2_flt, float v2_flt, int cols, int read_offset_, uint reduction);

#endif /*KERNELS_H*/
