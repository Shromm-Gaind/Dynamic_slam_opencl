#ifndef KERNELS_H
#define KERNELS_H

// Declarations of local device functions used by the kernels.

// Tau_HSV_grad(I) := distance in 8D space [ sin(Hue ), cos(Hue ), Saturation , (Saturation)/dx , (Saturation)/dy , Value , (Value)/dx , (Vallue)/dy ]
float Tau_HSV_grad (float8 B, float8 c);

float8 Tau_HSV_grad_8chan (float8 B, float8 c);

///////////////////// Interpolation /////////////////////

float8 nearest_neigbour (float8* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);

float8 bilinear (__global float8* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);

float8 bilinear_SE3_grad (__global float8* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction, int i, int mm_pixels);

#endif /*KERNELS_H*/
