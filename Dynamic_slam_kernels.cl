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
//#pragma OPENCL EXTENSION cl_khr_fp16 : enable
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

#define MAX_INV_DEPTH		0	// fp32_params indices
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


__kernel void cvt_color_space(
	__global	uchar*	base,			//0
	__global	float4*	img,			//1
	__constant	uint*	uint_params		//2
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
	
	float V = max(R_float, max(G_float, B_float) ); 
	
	float min_rgb =	min(R_float, min(G_float, B_float));
	float divisor = V - min_rgb;
	
	float S = (V!=0)*(V-min_rgb)/V;
	
	float H = (   (V==R_float && V!=0)* (60*(G_float-B_float) / divisor )      \
	 +     (V==G_float && V!=0)*( 120 + (60*(B_float-R_float) / divisor ))	\
	 +     (V==B_float && V!=0)*( 240 + (60*(R_float-G_float) / divisor ))	\
	 ) / 360;
	
	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	uint img_row	= base_row + margin;
	uint img_col	= base_col + margin;
	uint img_index	= img_row*mm_cols + img_col;   
	
	float4 temp_float4  = {H,S,V,0};											// Note how to load a float4 vector.
	img[img_index   ] = temp_float4;

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
/*
__kernel void mipmap(
	__global	float4*	img,			//0
	__global	float* 	gaussian,		//1
	__constant	uint*	uint_params,	//2
	__constant	uint*	mipmap_params,	//3
	__local		float4*	local_img_patch //4
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
	uint mm_rows		= uint_params[MM_ROWS];
	
	uint read_row    	= global_id/write_cols_; 																					// NB importance of integer division)
	read_row			*=2;
	uint read_column 	= 2*fmod(global_id,write_cols_);
	uint write_row    	= global_id/write_cols_;
	uint write_column 	= fmod(global_id,write_cols_);
	uint read_index 	= read_offset_  +  read_row*mm_cols  + read_column ;
	uint write_index 	= write_offset_ +  write_row*mm_cols + write_column ;
	float4 temp_float4;
	
	if (global_id < mipmap_params[MiM_PIXELS]) {
		for (int i=0; i<3; i++){																									// Load local_img_patch
			if ( (1+ lid + i*patch_length) >  49152/16      ||   read_index +i*mm_cols    >= mm_cols*mm_rows )  return;
			temp_float4 = img[read_index +i*mm_cols]; 
			local_img_patch[1+ lid + i*patch_length/2] = temp_float4;
			//local_img_patch[1+ lid + i*patch_length] = img[read_index +i*mm_cols];
		}
		if ((lid==0)||(lid==group_size-1)){
			int step = (lid==group_size-1)*(patch_length-1);
			for (int i=0; i<3; i++){
				int patch_index = step + i*patch_length;
				int read_index_2 = read_index -3*(lid==0) + 6*(lid==group_size-1) +(i*mm_cols)*3;
				float4 temp_float4 = img[read_index_2];
				local_img_patch[patch_index] = temp_float4;//img[read_index_2];//temp_half4;
	}	}	}
	barrier(CLK_LOCAL_MEM_FENCE);																									// fence between write & read local mem
	float4 reduced_pixel4 =0;																										// Gaussian blurr
	uint patch_read_index4 	 = lid;
	//uint patch_read_index4_2 = 0;
	uint index =0;
	for (int i=0; i<gaussian_size_; i++){ 
		for (int j=0; j<gaussian_size_; j++){
			index = patch_read_index4 +j;
			reduced_pixel4 += local_img_patch[index] ;// * gaussian[ j + gaussian_size_*i ];
		}	
		patch_read_index4   += patch_length ;
		//patch_read_index4_2 += patch_length ;
	}
	if (global_id > mipmap_params[MiM_PIXELS]) return;
	//reduced_pixel4.y  = lid;//((float)lid)/(5*(float)group_size);
	//reduced_pixel4.z  = read_index + mm_cols;//global_id*(640/480)/((float)(write_cols_*write_cols_));
	reduced_pixel4.w  = 1.0;
	
	img[ write_index ]	= reduced_pixel4;
	//uint sample = 192; // (global_id_u<sample+74 && global_id_u>sample )
	//if ((lid<4 || lid > 252 ) && global_id_u<2000) printf("\n  global_id_u=%u,  lid=%u, read_index=%i, %i,   mm_cols=%u,   index=%u,  write_index=%i, %i,  read_column=%u,  write_cols_=%u,  read_offset_=%u,  read_row=%u,  mm_cols=%u, reduced_pixel4.x=%f,  reduced_pixel4.z=%f,  reduced_pixel4.y=%f, ### temp_float4=(%f,%f,%f,%f) ", \
	//	global_id_u, lid, (read_index-read_offset_)/2, read_index, mm_cols, index, write_index-write_offset_, write_index, read_column, write_cols_, read_offset_,  read_row,  mm_cols, reduced_pixel4.x, reduced_pixel4.z, reduced_pixel4.y, temp_float4.x, temp_float4.y, temp_float4.z, temp_float4.w );
}
*/

__kernel void mipmap_flt(																											// Nvidia Geforce GPUs cannot use "half"
	__global float4*	img,			//0
	__global float* 	gaussian,		//1
	__global uint*		uint_params,	//2
	__global uint*		mipmap_params,	//3
	__local	 float4*	local_img_patch //4
		 )
{
	float global_id 	= get_global_id(0);
	if (global_id > mipmap_params[MiM_PIXELS]) return;
	
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	uint patch_length	= group_size+2;
	
	uint read_offset_ 	= 1*mipmap_params[MiM_READ_OFFSET];
	uint write_offset_ 	= 1*mipmap_params[MiM_WRITE_OFFSET];
	uint read_cols_ 	= mipmap_params[MiM_READ_COLS];
	uint write_cols_ 	= mipmap_params[MiM_WRITE_COLS];
	uint gaussian_size_ = mipmap_params[MiM_GAUSSIAN_SIZE];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	
	uint read_row    = 2*global_id/write_cols_;
	uint read_column = 2*fmod(global_id,write_cols_);
	
	uint write_row    = global_id/write_cols_;
	uint write_column = fmod(global_id,write_cols_);
	
	uint read_index 	= read_offset_  + 1*( read_row*mm_cols  + read_column  );	// NB 4 channels.
	uint write_index 	= write_offset_ + 1*( write_row*mm_cols + write_column );
	
	for (int i=0; i<3; i++){																										// Load local_img_patch
		local_img_patch[lid+1 + i*patch_length] = img[ read_index +i*mm_cols];
	}
	if (lid==0){
		for (int i=0; i<3; i++){
			local_img_patch[lid + i*patch_length] = img[ read_index +i*mm_cols -1];
		}
	}
	if (lid==group_size-1){
		for (int i=0; i<3; i++){
			local_img_patch[lid+2 + i*patch_length] = img[ read_index +i*mm_cols +1];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);																									// fence between write & read local mem
	float4 reduced_pixel = 0;
	for (int i=0; i<3; i++){
		for (int y=0; y<3; y++){
			reduced_pixel += local_img_patch[lid+y + i*patch_length]/9;																// 3x3 box filter, rather than Gaussian
		}
	}
	img[ write_index] = reduced_pixel;
}


__kernel void  img_grad(
	__global 	float4*	img,			//0 
	__constant 	uint*	uint_params,	//1
	__constant 	float*	fp32_params,	//2
	__global 	float*	gxp,			//3
	__global 	float*	gyp,			//4
	__global 	float*	g1p,			//5	 ?single channel  or tripple channel gradients ? prob triple given HSV.
	
	__global 	float4*	img_sum				//6
		 )
{
	 uint global_id_u = get_global_id(0);
	 int  x = global_id_u;
	 
	 if (x > uint_params[MM_PIXELS]) return;
	 
	 float4 img_pvt = img[x];		// required to make data in float4 accessible the kernel. 
	 //img_sum[x] = img_private;
	 
	 int rows 			= uint_params[MM_ROWS]  ;//floor(params[rows_]);
	 int cols 			= uint_params[MM_COLS]  ;//floor(params[cols_]);
	 float alphaG		= fp32_params[ALPHA_G];
	 float betaG 		= fp32_params[BETA_G];

	 int y = x / cols;
	 x = x % cols;
	 
	 int min = 10*cols + cols/2;
	 if((global_id_u<min+10)&&(global_id_u>min)){
		 printf("\n global_id_u=1,  rows=%i,  cols=%i,   x=%i,  y=%i,  img_pvt[x]=(%f,%f,%f,%f) ###########", rows, cols, x, y, img_pvt.x, img_pvt.y, img_pvt.z, img_pvt.w );
		 //img[x].x, img[x].y, img[x].z, img[x].w 
	 }
	 
	 if (x<2 || x > cols-2 || y<2 || y>rows-2) return;  // needed for wider kernel
	 
	 int upoff = -(y != 0)*cols;                   // up, down, left, right offsets, by boolean logic.
	 int dnoff = (y < rows-1) * cols;
     int lfoff = -(x != 0);
	 int rtoff = (x < cols-1);

	 unsigned int offset = x + y * cols;

	 float4 pu, pd, pl, pr;                         // rho, photometric difference: up, down, left, right, of grayscale ref image.
	 float g0x, g0y, g0, g1;
	 
	 // need to load these to local patch ? or something else?
/*	 
	 pr =  img_pvt[offset + rtoff];// + base[offset + rtoff +1];	// replaced 'base' with 'img_pvt' NB 3chan, float4  // NB base = grayscale CV_8UC1 image.
	 pl =  img_pvt[offset + lfoff];// + base[offset + lfoff -1];
	 pu =  img_pvt[offset + upoff];// + base[offset + 2*upoff];
	 pd =  img_pvt[offset + dnoff];// + base[offset + 2*dnoff];

	 float gx, gy;
	 gx			= fabs(pr.z - pl.z);	// NB HSV color space.
	 gy			= fabs(pd.z - pu.z);
*/	 
	 g1p[offset]= img_pvt.w;	//exp(-alphaG * pow(sqrt(gx*gx + gy*gy), betaG) );
	 gxp[offset]= img_pvt.x;	//offset;//((float)(offset))/((float)(uint_params[MM_PIXELS]));	//gx;   offset= column
	 gyp[offset]= img_pvt.y;	//x;//((float)(x))/((float)(uint_params[MM_PIXELS]));		//gy;
	 
	 
	 if(global_id_u==1)printf("\n global_id_u=1 ###########");//printf("\n (x=%i,img(%f,%f,%f,%f)), ", x, img[x].x, img[x].y, img[x].z, img[x].w );  //(fmod((float)(x),1000)==0)
}



/*
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

