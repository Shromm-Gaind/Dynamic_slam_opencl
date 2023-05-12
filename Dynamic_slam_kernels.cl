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

#define MAX_INV_DEPTH		0	// fp32_params indices, 		for DTAM mapping algorithm.
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

#define PIXELS				0	// uint_params indices, 		when launching one kernel per layer. 	Constant throughout program run.
#define ROWS				1	// baseimage
#define COLS				2
#define LAYERS				3
#define MARGIN				4
#define MM_PIXELS			5	// whole mipmap
#define MM_ROWS				6	
#define MM_COLS				7

#define MiM_PIXELS			0	// for mipmap_buf, 				when launching one kernel per layer. 	Updated for each layer.
#define MiM_READ_OFFSET		1	// for ths layer, 				start of image data
#define MiM_WRITE_OFFSET	2
#define MiM_READ_COLS		3	// cols without margins
#define MiM_WRITE_COLS		4
#define MiM_GAUSSIAN_SIZE	5	// filter box size


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
	
	 if (!(H<=1.0 && H>=0.0) || !(S<=1.0 && S>=0.0) || !(V<=1.0 && V>=0.0) ) {H=S=V=0;} ; // to replace any NaNs
	 
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

__kernel void cvt_color_space_linear(										// Writes the first entry in a linear mipmap
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
	
	 if (!(H<=1.0 && H>=0.0) || !(S<=1.0 && S>=0.0) || !(V<=1.0 && V>=0.0) ) {H=S=V=0;} ; // to replace any NaNs
	 
	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	uint img_row	= base_row + margin;
	uint img_col	= base_col + margin;
	uint img_index	= img_row*(cols + 2*margin) + img_col;   								// NB here use cols not mm_cols
	
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
	if (global_id > mipmap_params[MiM_PIXELS]) return;								// for Intel GPU all threads must reach barrier (above).
	img[ write_index] = reduced_pixel;
}

__kernel void mipmap_linear_flt(																							// Nvidia Geforce GPUs cannot use "half"
	__global float4*	img,			//0
	__global float* 	gaussian,		//1
	__global uint*		uint_params,	//2
	__global uint*		mipmap_params,	//3
	__local	 float4*	local_img_patch //4
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	uint patch_length	= group_size+2;
	
	uint read_offset_ 	= 1*mipmap_params[MiM_READ_OFFSET];
	uint write_offset_ 	= 1*mipmap_params[MiM_WRITE_OFFSET]; // = read_offset_ + read_cols_*read_rows for linear MipMap.
	
	uint read_cols_ 	= mipmap_params[MiM_READ_COLS];
	uint write_cols_ 	= mipmap_params[MiM_WRITE_COLS];
	
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];   // whole mipmap                       // = (read_cols_  + 2*margin ); // uint_params[MM_COLS];
	
	uint write_row   	= global_id_u / write_cols_ ;			//  global_id/write_cols_;	gentype fract(gentype x, __private gentype *iptr)
	uint write_column 	= fmod(global_id_flt, write_cols_);
	
	uint read_row    	= 2*write_row;
	uint read_column 	= 2*write_column;
	
	uint read_index 	= read_offset_  +  read_row  * (read_cols_+2*margin)  + read_column  ;		// NB 4 channels.
	uint write_index 	= write_offset_ +  write_row * (write_cols_+margin) + write_column ;
	
	for (int i=0; i<3; i++){																								// Load local_img_patch
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
	barrier(CLK_LOCAL_MEM_FENCE);																							// fence between write & read local mem
	float4 reduced_pixel = 0;
	for (int i=0; i<3; i++){
		for (int y=0; y<3; y++){
			reduced_pixel += local_img_patch[lid+y + i*patch_length]/9;														// 3x3 box filter, rather than Gaussian
		}
	}
	if (global_id_u > mipmap_params[MiM_PIXELS]) return;	// num pixels to b written & num threads to really use.				// for Intel GPU all threads must reach barrier (above).
	img[ write_index] = reduced_pixel;
	
	// (global_id==0 || global_id==1)
	if (write_column<2 && write_row<4) printf("\n global_id=%u, mipmap_params[MiM_PIXELS]=%u , read_index=%u , read_row=%u , read_column=%u , write_index=%u , write_row=%u , write_column=%u ,     read_offset_=%u , read_cols_=%u  ,  write_offset_=%u , write_cols_=%u , margin=%u , mm_cols=%u", \
												  global_id_u,    mipmap_params[MiM_PIXELS],     read_index ,    read_row  ,   read_column ,    write_index,     write_row ,    write_column ,        read_offset_ ,    read_cols_  ,     write_offset_ ,    write_cols_ ,    margin ,    mm_cols );
}


__kernel void  img_grad(
	__global 	float4*	img,			//0 
	__constant 	uint*	uint_params,	//1
	__constant 	float*	fp32_params,	//2
	__global 	float4*	gxp,			//3
	__global 	float4*	gyp,			//4
	__global 	float4*	g1p				//5
		 )
{
	 uint global_id_u = get_global_id(0);
	 int  x = global_id_u;
	 if (x > uint_params[MM_PIXELS]) return;
	 
	 float4 img_pvt 	= img[x];							// required to make data in float4 accessible the kernel. 
	 uint px_per_layer	= uint_params[ROWS] * uint_params[COLS];
	 uint layer_thresh	= px_per_layer;
	 int rows 			= uint_params[MM_ROWS];
	 int cols 			= uint_params[MM_COLS];
	 float alphaG		= fp32_params[ALPHA_G];
	 float betaG 		= fp32_params[BETA_G];

	 
	 bool level[5];
	 uint layer_offset = 0;
	 
	 for (int i=1; i<5; i++){
		 if(global_id_u < layer_thresh) break;
		 layer_offset	+=layer_thresh;
		 rows			/=2;
		 cols			/=2;
		 px_per_layer	/=2;
		 layer_thresh	+=px_per_layer;
	 }
	 x -= layer_offset;
	 int y = x / cols;
	 x = x % cols;
	 int min = 10*cols + cols/2;
	 if (x<2 || x > cols-2 || y<2 || y>rows-2) return;  	// needed for wider kernel
	 
	 int upoff		= -(y != 0)*cols;						// up, down, left, right offsets, by boolean logic.
	 int dnoff		= (y < rows-1) * cols;
     int lfoff		= -(x != 0);
	 int rtoff		= (x < cols-1);
	 uint offset	= x + y * cols + layer_offset;
	 
	 float4 pu, pd, pl, pr;									// rho, photometric difference: up, down, left, right, of grayscale ref image.
	 pr =  img[offset + rtoff];								// replaced 'base' with 'img_pvt' NB 3chan, float4  // NB base = grayscale CV_8UC1 image.
	 pl =  img[offset + lfoff];
	 pu =  img[offset + upoff];
	 pd =  img[offset + dnoff];

	 float4 gx	= { fabs(pr.x - pl.x), fabs(pr.y - pl.y), fabs(pr.z - pl.z), 1.0 };	// NB HSV color space.
	 float4 gy	= { fabs(pd.x - pu.x), fabs(pd.y - pu.y), fabs(pd.z - pu.z), 1.0 };
	 
	 float4 g1  = { \
		 exp(-alphaG * pow(sqrt(gx.x*gx.x + gy.x*gy.x), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.y*gx.y + gy.y*gy.y), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.z*gx.z + gy.z*gy.z), betaG) ), \
		 1.0 };
		 
	 g1p[offset]= g1;
	 gxp[offset]= gx;
	 gyp[offset]= gy;
}


__kernel void comp_param_maps(
	__constant 	uint*	uint_params,	//0
	__constant 	float*	fp32_params,	//1
	__global 	uint*	mipmap_params,	//2
	__global 	float* 	k2k,			//3
	__global 	float* 	depth_map,		//4
	__global 	float*	param_map		//5
		 )
{
	float global_id 	= get_global_id(0);
	if (global_id > mipmap_params[MiM_PIXELS]) return;
	
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	uint patch_length	= group_size+2;
	
	uint read_offset_ 	= 1*mipmap_params[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params[MiM_READ_COLS];
	uint write_cols_ 	= mipmap_params[MiM_WRITE_COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	
	uint read_row		= 2*global_id/write_cols_;
	uint read_column	= 2*fmod(global_id,write_cols_);
	uint read_index 	= read_offset_  + 1*( read_row*mm_cols  + read_column  );	//TODO NB 4 channels.
	
// SE3 
	// Rotate 0.001 radians i.e 0.0573  degrees
	// Translate 0.001 'units' of distance 
	const float delta_theta = 0.001;
	const float delta 	  	= 0.001;
	const float cos_theta   = cos(delta_theta);
	const float sin_theta   = sin(delta_theta);
	
	const float Rx[9] = {1.0, 0.0, 0.0,					0.0, cos_theta, -sin_theta, 		0.0, sin_theta, cos_theta	};
	const float Ry[9] = {cos_theta, 0.0, sin_theta,		0.0, 1.0, 0.0, 						-sin_theta, 0, cos_theta	};
	const float Rz[9] = {cos_theta, -sin_theta, 0.0, 	sin_theta, cos_theta, 				0.0, 	0.0, 0.0, 1.0		};
	
	const float Tx[16] = {1,0,0,delta, 	0,1,0,0,		0,0,1,0,		0,0,0,1};
	const float Ty[16] = {1,0,0,0, 		0,1,0,delta,	0,0,1,0,		0,0,0,1};
	const float Tz[16] = {1,0,0,0, 		0,1,0,0,		0,0,1,delta,	0,0,0,1};
	
	// TODO
	// Move k2k generation to host.
	// Create a 'reproject' & 'img_grad_sum' kernels 
	
	
	
	
	
	float reduced_pixel; // TODO how many channels ?
	param_map[read_index] = reduced_pixel;
	
}


__kernel void se3_grad(
	__global 	float4*	img,			//0 
	__constant 	uint*	uint_params,	//1
	__constant 	float*	fp32_params,	//2
	__global 	float4*	gxp,			//3
	__global 	float4*	gyp,			//4
	__global 	float4*	g1p,			//5
	__constant 	float*	k2k				//6
	
		 )
{// find gradient wrt SE3 find global sum for each of the 6 DoF
	
	
	
	
	

}


/* From  BuildCostVolume2(
 * 	__global float* k2k,		//0
	__global float* base,		//1  // uchar*
	__global float* img,		//2  // uchar*
	__global float* cdata,		//3
	__global float* hdata,		//4  'w' num times cost vol elem has been updated
	__global float* lo, 		//5
	__global float* hi,			//6
	__global float* a,			//7
	__global float* d,			//8
	__constant float* params,	//9 pixels, rows, cols, layers, max_inv_depth, min_inv_depth, inv_d_step, threshold
	__global float* img_sum)	//10
	)
	{
	int global_id = get_global_id(0);

	int pixels 			= floor(params[pixels_]);
	int rows 			= floor(params[rows_]);
	int cols 			= floor(params[cols_]);
	int layers			= floor(params[layers_]);
	float max_inv_depth = params[max_inv_depth_];  // not used
	float min_inv_depth = params[min_inv_depth_];
	float inv_d_step 	= params[inv_d_step_];
	float u 			= global_id % cols;														// keyframe pixel coords
	float v 			= (int)(global_id / cols);
	int offset_3 		= global_id *3;															// Get keyframe pixel values
	float3 B;		B.x = base[offset_3];	B.y = base[offset_3];	B.z = base[offset_3];		// pixel from keyframe

	float 	u2,	v2, rho,	inv_depth=0.0,	ns=0.0,	mini=0.0,	minv=3.0,	maxv=0.0;			// variables for the cost vol
	int 	int_u2, int_v2, coff_00, coff_01, coff_10, coff_11, cv_idx=global_id,	layer = 0;
	float3 	c, c_00, c_01, c_10, c_11;
	float 	c0 = cdata[cv_idx];																	// cost for this elem of cost vol
	float 	w  = hdata[cv_idx];																	// count of updates of this costvol element. w = 001 initially

	// layer zero, ////////////////////////////////////////////////////////////////////////////////////////
	// inf depth, roation without paralax, i.e. reproj without translation.
	// Use depth=1 unit sphere, with rotational-preprojection matrix

	// precalculate depth-independent part of reprojection, h=homogeneous coords.
	float uh2 = k2k[0]*u + k2k[1]*v + k2k[2]*1;  // +k2k[3]/z
	float vh2 = k2k[4]*u + k2k[5]*v + k2k[6]*1;  // +k2k[7]/z
	float wh2 = k2k[8]*u + k2k[9]*v + k2k[10]*1; // +k2k[11]/z
	//float h/z  = k2k[12]*u + k2k[13]*v + k2k[14]*1; // +k2k[15]/z
	float uh3, vh3, wh3;

	// cost volume loop  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	#define MAX_LAYERS 256 //64
	float cost[MAX_LAYERS];
	for( layer=0;  layer<layers; layer++ ){
		cv_idx = global_id + layer*pixels;
		cost[layer] = cdata[cv_idx];													// cost for this elem of cost vol
		w  = hdata[cv_idx];																// count of updates of this costvol element. w = 001 initially
		inv_depth = (layer * inv_d_step) + min_inv_depth;								// locate pixel to sample from  new image. Depth dependent part.
		uh3  = uh2 + k2k[3]*inv_depth;
		vh3  = vh2 + k2k[7]*inv_depth;
		wh3  = wh2 + k2k[11]*inv_depth;
		u2   = uh3/wh3;
		v2   = vh3/wh3;
		//if(u==70 && v==470)printf("\n(inv_depth=%f,   ",inv_depth);
		int_u2 = ceil(u2-0.5);		// nearest neighbour interpolation
		int_v2 = ceil(v2-0.5);

		if ( !((int_u2<0) || (int_u2>cols-1) || (int_v2<0) || (int_v2>rows-1)) ) {  	// if (not within new frame) skip
			c=img[(int_v2*cols + int_u2)*3];											// nearest neighbour interpolation
			float rx=(c.x-B.x); float ry=(c.y-B.y); float rz=(c.z-B.z);					// Compute photometric cost // L2 norm between keyframe & new frame pixels.
			rho = sqrt( rx*rx + ry*ry + rz*rz )*50;										//TODO make *50 an auto-adjusted parameter wrt cotrast in area of interest.
			cost[layer] = (cost[layer]*w + rho) / (w + 1);
			cdata[cv_idx] = cost[layer];  												// CostVol set here ###########
			hdata[cv_idx] = w + 1;														// Weightdata, counts updates of this costvol element.
			img_sum[cv_idx] += (c.x + c.y + c.z)/3;
		}
	}
	for( layer=0;  layer<layers; layer++ ){
		if (cost[layer] < minv) { 														// Find idx & value of min cost in this ray of cost vol, given this update.
			minv = cost[layer];															// NB Use array private to this thread.
			mini = layer;
		}
		maxv = fmax(cost[layer], maxv);
	}
*/


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

*/

