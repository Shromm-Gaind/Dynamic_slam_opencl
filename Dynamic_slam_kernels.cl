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

#define MiM_PIXELS			0	// for mipmap_buf
#define MiM_READ_OFFSET		1
#define MiM_WRITE_OFFSET	2
#define MiM_READ_COLS		3
#define MiM_WRITE_COLS		4
#define MiM_GAUSSIAN_SIZE	5


__kernel void cvt_color_space(												// basemem(CV_8UC3, RGB)->imgmem(CV16FC3, HSV) using OpenCL 'half'.
	__global uchar*		base,			//0
	__global half*		img,			//1									// NB half has approximately 3 decimal significat figures, and +/-5 decimal orders of magnitude
	__global uint*		uint_params		//2
		 )
{																			// NB need 32-bit uint (2**32=4,294,967,296) for index, not 16bit (2**16=65,536).
	int global_id 	= (int)get_global_id(0);
	uint pixels 	= uint_params[PIXELS];
	if (global_id > pixels) return;
	
	uint cols 		= uint_params[COLS];
	uint margin 	= uint_params[MARGIN];
	uint mm_cols	= uint_params[MM_COLS];

	float R_float	= base[global_id*3]  /256.0f;
	float G_float	= base[global_id*3+1]/256.0f;
	float B_float	= base[global_id*3+2]/256.0f;
	half  R,G,B, H,S,V;
	vstore_half(R_float, 0, &R);
	vstore_half(G_float, 0, &G);
	vstore_half(B_float, 0, &B);
	
	float V_max_f = max(R_float, max(G_float, B_float) );
	vstore_half( V_max_f, 0, &V );
	
	half min_rgb; 	vstore_half( min(R_float, min(G_float, B_float)) , 0, &min_rgb );
	half divisor = V - min_rgb;
	
	S = (V!=0)*(V-min_rgb)/V;
	
	H = (   (V==R && V!=0)*half_divide( (60*(G-B) ), divisor )      \
	 +     (V==G && V!=0)*( 120 + half_divide(60*(B-R), divisor ))	\
	 +     (V==B && V!=0)*( 240 + half_divide(60*(R-G), divisor ))	\
	 ) / 360;
	
	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	uint img_row	= base_row + margin;
	uint img_col	= base_col + margin;
	uint img_index	= img_row*mm_cols*3 + img_col*3;   
	img[img_index   ] = H;
	img[img_index +1] = S;
	img[img_index +2] = V;
	/* from https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
	 * In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and scaled to fit the 0 to 1 range.
	 * V = max(R,G,B)
	 * S = (0, if V=0), otherwise (V-min(RGB))/V
	 * H = 60(G-B)/(V-min(R,G,B)  			if V=R
	 *     120 + 60(B-R)/(V-min(R,G,B))		if V=G
	 *     240 + 60(R-G)/(V-min(R,G,B))		if V=B
	 *     0								if R=G=B
	 */
}

__kernel void mipmap(
	__global half*		img,			//0						// NB half has approximately 3 decimal significat figures, and +/-5 decimal orders of magnitude
	__global half* 		gaussian,		//1
	__global uint*		uint_params,	//2
	__global uint*		mipmap_params,	//3
	__local  half4*		local_img_patch //4
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id		= global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	uint patch_length	= group_size+2;
	
	uint read_offset_ 	= mipmap_params[MiM_READ_OFFSET];
	uint write_offset_ 	= mipmap_params[MiM_WRITE_OFFSET];
	uint write_cols_ 	= mipmap_params[MiM_WRITE_COLS];
	uint gaussian_size_ = mipmap_params[MiM_GAUSSIAN_SIZE];
	
	uint mm_cols		= uint_params[MM_COLS];
	
	uint read_row    	= 2*global_id/write_cols_;
	uint read_column 	= 2*fmod(global_id,write_cols_);
	uint write_row    	= global_id/write_cols_;
	uint write_column 	= fmod(global_id,write_cols_);
	uint read_index 	= read_offset_  + 3*( read_row*mm_cols  + read_column  );													// for 'img[..]'NB 3 channels.
	uint write_index 	= write_offset_ + 3*( write_row*mm_cols + write_column );

	for (int i=0; i<3; i++){
		//if(global_id_u == 12*write_cols_ + 10){
		half4 temp_half4 = {img[read_index +i*mm_cols*3], img[read_index+1 +i*mm_cols*3], img[read_index+2 +i*mm_cols*3], lid };	// Note how to load a half4 vector.
		local_img_patch[1+ lid + i*patch_length] = temp_half4;
	}
	if ((lid==0)||(lid==group_size-1)){
		int step = (lid==group_size-1)*(patch_length-1);
		for (int i=0; i<3; i++){
			int patch_index = step + i*patch_length;
			int read_index_2 = read_index -3*(lid==0) + 6*(lid==group_size-1) +(i*mm_cols)*3;
			half4 temp_half4 = {img[read_index_2], img[read_index_2 +1], img[read_index_2 +2], -1 };
			local_img_patch[patch_index] = temp_half4;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	half4 reduced_pixel4 =0;																										// Gaussian blurr
	uint patch_read_index4 	 = lid;
	uint patch_read_index4_2 = 0;
	for (int i=0; i<gaussian_size_; i++){ 
		for (int j=0; j<gaussian_size_; j++){
			int index = patch_read_index4 +j;
			reduced_pixel4 += local_img_patch[index] * gaussian[ j + gaussian_size_*i ];// 
		}	
		patch_read_index4   += patch_length ;
		patch_read_index4_2 += patch_length ;
	}
	if (global_id > mipmap_params[MiM_PIXELS]) return;
	img[ write_index+0 ] =	reduced_pixel4.x;
	img[ write_index+1 ] =	reduced_pixel4.y;
	img[ write_index+2 ] =	reduced_pixel4.z;
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

