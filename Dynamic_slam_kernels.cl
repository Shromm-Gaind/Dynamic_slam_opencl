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
#define MM_PIXELS			5
#define MM_ROWS				6	
#define MM_COLS				7


__kernel void cvt_color_space(	// basemem(CV_8UC3, RGB)->imgmem(CV16FC3, HSV) using OpenCL 'half'.
	__global uchar*		base,			//0
	__global half*		img,			//1
	__global uint*		uint_params,	//2
	__global float*		img_sum,		//3
	__global half*		half_params		//4
		 )
{																			// NB need 32-bit uint (2**32=4,294,967,296) for index, not 16bit (2**16=65,536).
	int global_id 	= (int)get_global_id(0);
	uint pixels 	= uint_params[PIXELS];
	
	if (global_id > pixels) return;
	
	uint rows		= uint_params[ROWS];
	uint cols 		= uint_params[COLS];
	uint margin 	= uint_params[MARGIN];
	uint mm_cols	= uint_params[MM_COLS];
	
	float float_params[11];
	for (int i=0; i<12; i++) float_params[i] = vload_half( i, half_params );
	
	if (global_id==0){ 
		printf("\n## global_id==1 ##, pixels=%u, rows=%u, cols=%u, margin=%u, mm_cols=%u, \n", pixels,rows,cols,margin,mm_cols);
		printf("\n## half_params, MAX_INV_DEPTH=%f,  SCALE_EAUX=%f\n", float_params[MAX_INV_DEPTH], float_params[SCALE_EAUX]);
	}
	
	uchar R_uchar	= base[global_id*3];
	uchar G_uchar	= base[global_id*3+1];
	uchar B_uchar	= base[global_id*3+2];
	
	uchar3 pixel 	= (uchar3)(R_uchar, G_uchar, B_uchar);
	float3 pixelf	= (float3)(pixel.x ,pixel.y, pixel.z);
	
	//half3 RGB;
	half  R,G,B, H,S,V;
	vstore_half(pixelf.x/256, 0, &R);
	vstore_half(pixelf.y/256, 0, &G);
	vstore_half(pixelf.z/256, 0, &B);
	
	uchar V_max   = max(R_uchar, max(G_uchar,B_uchar) );
	float V_max_f = ((float)V_max)/256;
	vstore_half( V_max_f, 0, &V );
	
	half min_rgb; vstore_half( ((float)min(R_uchar, min(G_uchar,B_uchar))/256 ), 0, &min_rgb );
	half divisor; vstore_half( (((float)V)/256)-min_rgb, 0, &divisor );
	
	S = (V!=0)*(V-min_rgb)/V;
	
	H = (V==R && V!=0)*native_divide( (60*(G-B) ), divisor )   \
	 +     (V==G && V!=0)*native_divide( 120 + 60*(B-R), divisor )	\
	 +     (V==B && V!=0)*native_divide( 240 + 60*(R-G), divisor )	;
	
	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	
	uint img_row	= base_row + margin;
	uint img_col	= base_col + margin;
	
	uint img_index	= img_row*mm_cols*3 + img_col*3;   
	//half3 img_pixel = (half3)(H,S,V);
	//img[img_index]	= img_pixel;
	
	
	//half3 img_pixel = (half3)(0.5, 0.5, 0.0);
	//img[global_id*3]	= img_pixel;
	img[img_index   ] = R;
	img[img_index +1] = G;
	img[img_index +2] = B;
	
	float Rf, Bf, Gf, Hf, Sf, Vf;
	Rf = vload_half(  0, &R );
	Gf = vload_half(  0, &G );
	Bf = vload_half(  0, &B );
	Hf = vload_half(  0, &H );
	Sf = vload_half(  0, &S );
	Vf = vload_half(  0, &V );
	
	//float3 img_pixel_f = (float3)(Rf, Bf, Gf);
	//img_sum[img_index*3] = img_pixel_f/256;
	
	img_sum[img_index]    =  Hf;//((float)pixel.x)/256;
	img_sum[img_index +1] =  Sf;//((float)pixel.y)/256;
	img_sum[img_index +2] =  Vf;//((float)pixel.z)/256;
	
	if (global_id==pixels-1) printf("\n## global_id==%u, img_index=%u, img_row=%u, img_col=%u ##", pixels-1, img_index, img_row, img_col);
	if (global_id==1000) printf("\n\nRf=%f , Gf=%f  , Bf=%f  , Hf=%f  , Sf=%f  ,  Vf=%f   \n\n",Rf, Gf, Bf, Hf, Sf, Vf );
	//if (global_id==1000) printf("\n\nR=%hx , G=%hx  , B=%hx  , H=%hx  , S=%hx  ,  V=%hx   \n\n",(short)R, (short)G, (short)B, (short)H, (short)S, (short)V );
	
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

