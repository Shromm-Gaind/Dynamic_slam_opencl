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
	__global half*		img,			//1		// NB half has approximately 3 decimal significat figures, and +/-5 decimal orders of magnitude
	__global uint*		uint_params//,	//2
	//__global float*		img_sum,		//3
	//__global half*		fp16_params		//4
		 )
{																			// NB need 32-bit uint (2**32=4,294,967,296) for index, not 16bit (2**16=65,536).
	int global_id 	= (int)get_global_id(0);
	uint pixels 	= uint_params[PIXELS];
	if (global_id > pixels) return;
	
	uint cols 		= uint_params[COLS];
	uint margin 	= uint_params[MARGIN];
	uint mm_cols	= uint_params[MM_COLS];
	/*																		// Testing cv::fp16 and opencl 'half'
	uint rows		= uint_params[ROWS];
	float float_params[11];
	//for (int i=0; i<12; i++) float_params[i] = vload_half( i, half_params );
	half  test_half1  = 1210;
	half  test_half2  = 10.201;
	half  test_half3  = 1.020;
	float test_float1 = vload_half(0, &test_half1);
	float test_float2 = vload_half(0, &test_half2);
	float test_float3 = vload_half(0, &test_half3);
	
	if (global_id==0){ 
		for (int i=0; i<11; i++){
			printf("\nhalf_params[%d]=%f,\t\t fp16_params[%d]=%f",i, half_params[i], i, fp16_params[i]);
		}
		printf("\n\n## global_id==1 ##, pixels=%u, rows=%u, cols=%u, margin=%u, mm_cols=%u, \n", pixels,rows,cols,margin,mm_cols);
		//printf("\n## half_params, MAX_INV_DEPTH=%hu,  SCALE_EAUX=%hu\n", half_params[MAX_INV_DEPTH], half_params[SCALE_EAUX]);
		//printf("\n## test_half1=%hu, test_half2=%hu, test_half3=%hu\n", test_half1, test_half2, test_half3); // 248 for all of them
		//printf("\n## test_half1=%hx, test_half2=%hx, test_half3=%hx\n", test_half1, test_half2, test_half3); // f8  for all of them 
		//printf("\n## test_half1= %hf, test_half2= %hf, test_half3= %hf\n", test_half1, test_half2, test_half3); // correct output
		printf("\n## test_half1= %f, test_half2= %f, test_half3= %f\n", test_half1, test_half2, test_half3); // correct output
		//printf("\n## float_params, MAX_INV_DEPTH=%f,  SCALE_EAUX=%f\n", float_params[MAX_INV_DEPTH], float_params[SCALE_EAUX]);
		printf("\n## test_float1= %f, test_float2= %f, test_float3= %f\n", test_float1, test_float2, test_float3); // correct output
		printf("\n## test_half1*fp16_params[0]= %f, test_half2+fp16_params[4]= %f, test_half3/fp16_params[3]= %f\n",\
			test_half1*fp16_params[0], test_half2+fp16_params[4], test_half3/fp16_params[3]); 	// verify that 'half' aritmetic works with cv::fp16 data. 
	}
	*/
	float R_float	= base[global_id*3]  /256.0;
	float G_float	= base[global_id*3+1]/256.0;
	float B_float	= base[global_id*3+2]/256.0;
	half  R,G,B, H,S,V;
	vstore_half(R_float, 0, &R);
	vstore_half(G_float, 0, &G);
	vstore_half(B_float, 0, &B);
	
	float V_max_f = max(R_float, max(G_float, B_float) );
	vstore_half( V_max_f, 0, &V );
	
	half min_rgb; 	vstore_half( min(R_float, min(G_float, B_float)) , 0, &min_rgb );
	half divisor = V - min_rgb;
	
	S = (V!=0)*(V-min_rgb)/V;
	
	H = (V==R && V!=0)*native_divide( (60*(G-B) ), divisor )   \
	 +     (V==G && V!=0)*native_divide( 120 + 60*(B-R), divisor )	\
	 +     (V==B && V!=0)*native_divide( 240 + 60*(R-G), divisor )	;
	
	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	uint img_row	= base_row + margin;
	uint img_col	= base_col + margin;
	uint img_index	= img_row*mm_cols*3 + img_col*3;   
	img[img_index   ] = H;
	img[img_index +1] = S;
	img[img_index +2] = V;
	/*
	 * //Debugging FP16
	float Rf, Bf, Gf, Hf, Sf, Vf;
	Rf = vload_half(  0, &R );
	Gf = vload_half(  0, &G );
	Bf = vload_half(  0, &B );
	Hf = vload_half(  0, &H );
	Sf = vload_half(  0, &S );
	Vf = vload_half(  0, &V );
	//uchar3 pixel 	= (uchar3)(R_uchar, G_uchar, B_uchar);
	img_sum[img_index]    =  Hf;//((float)pixel.x)/256;
	img_sum[img_index +1] =  Sf;//((float)pixel.y)/256;
	img_sum[img_index +2] =  Vf;//((float)pixel.z)/256;
	
	if (global_id==pixels-1) printf("\n## global_id==%u, img_index=%u, img_row=%u, img_col=%u ##", pixels-1, img_index, img_row, img_col);
	if (global_id==1000) printf("\n\nRf=%f , Gf=%f  , Bf=%f  , Hf=%f  , Sf=%f  ,  Vf=%f   \n\n",Rf, Gf, Bf, Hf, Sf, Vf );
	//if (global_id==1000) printf("\n\nR=%hx , G=%hx  , B=%hx  , H=%hx  , S=%hx  ,  V=%hx   \n\n",(short)R, (short)G, (short)B, (short)H, (short)S, (short)V );
	*/
	/* 
	 * from https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
	 * In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and scaled to fit the 0 to 1 range.
	 * V = max(R,G,B)
	 * S = (0, if V=0), otherwise (V-min(RGB))/V
	 * H = 60(G-B)/(V-min(R,G,B)  			if V=R
	 *     120 + 60(B-R)/(V-min(R,G,B))		if V=G
	 *     240 + 60(R-G)/(V-min(R,G,B))		if V=B
	 *     0								if R=G=B
	 */
}

__kernel void mipmap(// ? can CPU generate FP16 in wchar ? Could be fasterthan taking GPU space. Use cv::float16_t class to fill arrays.
	__global half*		img,			//0		// NB half has approximately 3 decimal significat figures, and +/-5 decimal orders of magnitude
	__global uint* 		read_offset,	//1
	__global uint* 		write_offset,	//2
	__global uint* 		read_cols,		//3
	__global uint* 		write_cols,		//4
	__global uint* 		pixels,			//5		// num pixels in reduced image &=> threads req.
	__global uint* 		gaussian_size,	//6
	__global half* 		gaussian,		//7
	__global uint*		uint_params//,	//8
		 )
{
	int global_id 		= (int)get_global_id(0);
	uint pixels_ = *pixels;
	if (global_id > pixels_) return;
	
	uint read_offset_ 	= *read_offset;
	uint write_offset_ 	= *write_offset;
	uint read_cols_ 	= *read_cols;
	uint write_cols_ 	= *write_cols;
	uint gaussian_size_ = *gaussian_size;
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint read_index 	= 3*(read_offset_  + (global_id/read_cols_ )*mm_cols + (global_id%read_cols_ ) );	// NB 3 channels.
	uint write_index 	= 3*(write_offset_ + (global_id/write_cols_)*mm_cols + (global_id%write_cols_) );
	
	half reduced_pixel_x=0, reduced_pixel_y=0, reduced_pixel_z=0;
	for (int i=0; i<gaussian_size_; i++){ for (int j=0; j<gaussian_size_; j++){
			reduced_pixel_x += img[ read_index+0 + 3*(j + mm_cols*i) ] * gaussian[ j + gaussian_size_*i ];
			reduced_pixel_y += img[ read_index+1 + 3*(j + mm_cols*i) ] * gaussian[ j + gaussian_size_*i ];
			reduced_pixel_z += img[ read_index+2 + 3*(j + mm_cols*i) ] * gaussian[ j + gaussian_size_*i ];
	}	}
	// TODO NB this would be better for split color mipmap BUT any vector of 3 would be stored on a vector of 4, as a power of 2.
	img[ write_index+0 ] = reduced_pixel_x;		
	img[ write_index+1 ] = reduced_pixel_y;
	img[ write_index+2 ] = reduced_pixel_z;
}

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

