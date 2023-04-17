/*
// the data in float p[] , set in RunCL.cpp,    void RunCL::calcCostVol(...),   variables declared in RunCL.h  class RunCL{...}
//  res = clSetKernelArg(cost_kernel, 0, sizeof(cl_mem),    &k2kbuf);
//  res = clSetKernelArg(cost_kernel, 1, sizeof(cl_mem),    &basemem);
//  res = clSetKernelArg(cost_kernel, 2, sizeof(cl_mem),    &imgmem);
//  res = clSetKernelArg(cost_kernel, 3, sizeof(cl_mem),    &cdatabuf);
//  res = clSetKernelArg(cost_kernel, 4, sizeof(cl_mem),    &hdatabuf);

//  res = clSetKernelArg(cost_kernel, 5, sizeof(cl_mem),    &lomem);
//  res = clSetKernelArg(cost_kernel, 6, sizeof(cl_mem),    &himem);
//  res = clSetKernelArg(cost_kernel, 7, sizeof(cl_mem),    &amem);
//  res = clSetKernelArg(cost_kernel, 8, sizeof(cl_mem),    &dmem);
//  res = clSetKernelArg(cost_kernel, 9, sizeof(int),       &param_buf);
*/

//#pragma OPENCL EXTENSION <extension_name> : <behavior>
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
/*
#define pixels_			0  // fp16_params indices
#define rows_			1
#define cols_			2
#define layers_			3
#define max_inv_depth_	4
#define min_inv_depth_	5
#define inv_d_step_		6
#define alpha_g_		7
#define beta_g_			8	///  __kernel void CacheG4
#define epsilon_		9	///  __kernel void UpdateQD		// epsilon = 0.1
#define sigma_q_		10									// sigma_q = 0.0559017
#define sigma_d_		11
#define theta_			12
#define lambda_			13	///   __kernel void UpdateA2
#define scale_Eaux_		14
#define margin_			15
*/

#define MAX_INV_DEPTH		0	// fp16_params indices
#define MIN_INV_DEPTH		1
#define INV_DEPTH_STEP		2
#define ALPHA_G				3
#define BETA_G				4	//  __kernel void CacheG4
#define EPSILON 			5	//  __kernel void UpdateQD		// epsilon = 0.1
#define SIGMA_Q 			6									// sigma_q = 0.0559017
#define SIGMA_D 			7
#define THETA				8
#define LAMBDA				9	//  __kernel void UpdateA2
#define SCALE_EAUX			10

#define PIXELS				0	// uint_params indices
#define ROWS				1	
#define COLS				2
#define LAYERS				3
#define MARGIN				4


__kernel void cvt_color_space(	// basemem(CV_8UC3, RGB)->imgmem(CV16FC3, HSV) using OpenCL 'half'.
	__global uchar3*	base,			//0
	__global half3*		img,			//1
	__global uint*		uint_params		//2
	//__global half*		fp16_params		//4
		 )
{																			// NB need 32-bit uint (2**32=4,294,967,296) for index, not 16bit (2**16=65,536).
	uint global_id 	= get_global_id(0);
	uint pixels 	= uint_params[PIXELS];
	if (global_id > pixels) return;
	
	uint rows		= uint_params[ROWS];
	uint cols 		= uint_params[COLS];
	uint margin 	= uint_params[MARGIN];
	
	uchar3 pixel 	= base[global_id];
	uchar R_uchar 	= pixel.x;
	uchar G_uchar	= pixel.y;
	uchar B_uchar	= pixel.z;
	
	half R			= pixel.x;
	half G 			= pixel.y;
	half B 			= pixel.z;
	
	half V 			= max(R_uchar, max(G_uchar,B_uchar) );
	
	half min_rgb	= min(R_uchar, min(G_uchar,B_uchar) );
	half divisor	= V-min_rgb;
	
	half S = (V!=0)*(V-min_rgb)/V;
	
	half H = (V==R && V!=0)*native_divide( (60*(G-B) ), divisor )   \
	 +     (V==G && V!=0)*native_divide( 120 + 60*(B-R), divisor )	\
	 +     (V==B && V!=0)*native_divide( 240 + 60*(R-G), divisor )	;
	
	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	
	uint img_row	= base_row + margin;
	uint img_col	= base_col + margin;
	
	uint img_index	= img_row*(cols+(margin*4)) + img_col;   
	half3 img_pixel = (half3)(H,S,V);
	img[img_index]	= img_pixel;
	/* 
	 * from https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
	 * V = max(R,G,B)
	 * S = (0, if V=0), otherwise (V-min(RGB))/V
	 * H = 60(G-B)/(V-min(R,G,B)  			if V=R
	 *     120 + 60(B-R)/(V-min(R,G,B))		if V=G
	 *     240 + 60(R-G)/(V-min(R,G,B))		if V=B
	 *     0								if R=G=B
	 */
}
/*
__kernel void convertParams(// ? can CPU generate FP16 in wchar ? Could be fasterthan taking GPU space. Use cv::float16_t class to fill arrays.
	__global float* k2k,		//0
	__global half* 	k2k_half,	//1
	__global float* params,		//2
	__global half* 	params_half //3
		 )
{
	
	
}
*/
/*__kernel void  (
	__global float* k2k,		//0
	
		 )
{
	
	
}
__kernel void  (
	__global float* k2k,		//0
	
		 )
{
	
	
}
__kernel void  (
	__global float* k2k,		//0
	
		 )
{
	
	
}
__kernel void  (
	__global float* k2k,		//0
	
		 )
{
	
	
}
__kernel void  (
	__global float* k2k,		//0
	
		 )
{
	
	
}

*/

