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

/*  Test code for fluid.h for Morphogenesis opencl

struct demoStruct {
	float data_float;
};

inline float get_data(struct demoStruct obj  ){return obj.data_float;}

__constant struct demoStruct demoStructObj = {12.6};

//__constant float demo_float = get_data(demoStructObj);

////

struct FBufs {
	char*				mcpu[ MAX_BUF ];
	#ifdef OpenCL_KERNEL
		char*			mgpu[ MAX_BUF ];		// on device, pointer is local.
	#else			
		CUdeviceptr		mgpu[ MAX_BUF ];		// on host, gpu is a device pointer // an array of pointers, filled by cuMemAlloc
		CUdeviceptr		gpu    (int n )	{ return mgpu[n];  }
		CUdeviceptr*	gpuptr (int n )	{ return &mgpu[n]; }		
	#endif		
};
#ifdef OpenCL__KERNEL
			// on device, access data via gpu pointers 
			//inline CALLFUNC Vector3DF*           bufV3(int n)		 { return (Vector3DF*) mgpu[n]; }
			inline  float3*              bufF3(struct FBufs obj,  int n)		 { return (float3*) obj.mgpu[n]; }
			inline CALLFUNC float*               bufF (int n)		 { return (float*)  mgpu[n]; }
			inline CALLFUNC uint*                bufI (int n)		 { return (uint*)   mgpu[n]; }
			inline CALLFUNC char*                bufC (int n)		 { return (char*)   mgpu[n]; }
			inline CALLFUNC uint**               bufII (int n)       { return (uint**)  mgpu[n]; }               // for elastIdx[][]
			inline CALLFUNC curandState_t*       bufCuRNDST (int n)  { return (curandState_t*)  mgpu[n]; }
			inline CALLFUNC unsigned long long*  bufULL (int n)      { return (unsigned long long*)  mgpu[n]; }
#else
			// on host, access data via cpu pointers
			//inline CALLFUNC Vector3DF*           bufV3(int n)		 { return (Vector3DF*) mcpu[n]; }
			inline  float3*              bufF3(struct FBufs obj,  int n)		 { return (float3*) obj.mcpu[n]; }
			inline  float*               bufF (int n)		 { return (float*)  mcpu[n]; }
			inline  uint*                bufI (int n)		 { return (uint*)   mcpu[n]; }
			inline  char*                bufC (int n)		 { return (char*)   mcpu[n]; }
			inline  uint**               bufII (int n)       { return (uint**)  mcpu[n]; }               // for elastIdx[][]
			inline  curandState_t*       bufCuRNDST (int n)  { return (curandState_t*)  mcpu[n]; }
			inline  unsigned long long*  bufULL (int n)      { return (unsigned long long*)  mcpu[n]; }
#endif
*/

///////////////////

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
#define SE3_LM_A			11	// LM damped least squares parameters for SE3 tracking
#define SE3_LM_B			12

#define PIXELS				0	// uint_params indices, 		when launching one kernel per layer. 	Constant throughout program run.
#define ROWS				1	// baseimage
#define COLS				2
#define COSTVOL_LAYERS		3
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
#define MiM_READ_ROWS		6	// rows without margins
#define MiM_WRITE_ROWS		7

#define IMG_MEAN			0	// for img_stats
#define IMG_VAR 			1	//



__kernel void cvt_color_space_linear(																// Writes the first entry in a linear mipmap, and computes img_mean
	__global	uchar*	base,			//0															// NB for debugging the mimpam is arranged as a series below eachother with margins.
	__global	float4*	img,			//1															// This can be changed to dense packing in a linear array, to reduce memeory and data transfer requirements.
	__constant	uint*	uint_params,	//2
	__constant 	uint8*	mipmap_params,	//3
	__local		float4*	local_sum_pix,	//4
	__global	float4*	global_sum_pix	//5
		 )
{																									// NB need 32-bit uint (2**32=4,294,967,296) for index, not 16bit (2**16=65,536).
	int global_id 			= (int)get_global_id(0);
	uint pixels 			= uint_params[PIXELS];
	uint lid 				= get_local_id(0);
	uint local_size 		= get_local_size(0);
	uint group_size 		= local_size;
	uint reduction			= 1;																	// = mm_cols/read_cols_; but this is only the baselayer ofthe image pyramid.
	
	uint8 mipmap_params_	= mipmap_params[0];
	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
	uint cols 				= uint_params[COLS];
	//uint margin 			= uint_params[MARGIN];
	uint mm_cols			= uint_params[MM_COLS];

	float R_float			= base[global_id*3]  /256.0f;
	float G_float			= base[global_id*3+1]/256.0f;
	float B_float			= base[global_id*3+2]/256.0f;
	
	float V 				= max(R_float, max(G_float, B_float) ); 
	float min_rgb 			= min(R_float, min(G_float, B_float) );
	float divisor 			= V - min_rgb;
	float S 				= (V!=0)*(V-min_rgb)/V;
	/*
	float H = (   (V==R_float && V!=0)* 		(60*(G_float-B_float) / divisor )  \
			+     (V==G_float && V!=0)*( 120 + 	(60*(B_float-R_float) / divisor )) \
			+     (V==B_float && V!=0)*( 240 + 	(60*(R_float-G_float) / divisor )) \
			) / 360;
	*/
	// Hue in radians
	const float Pi_3 = M_PI_F/3;
	
	float H = (   (V==R_float && V!=0)* 	Pi_3* ((G_float-B_float) / divisor )  \
			+     (V==G_float && V!=0)*		Pi_3*(((B_float-R_float) / divisor ) +2) \
			+     (V==B_float && V!=0)*		Pi_3*(((R_float-G_float) / divisor ) +4) \
			);																				// TODO shift "/M_PI_F" to CPU data saving ?
	
	if (!(H<=2*M_PI_F && H>=0.0f) || !(S<=1.0f && S>=0.0f) || !(V<=1.0f && V>=0.0f) ) {H=S=V=0.0f;}		// to replace any NaNs
	
	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	//uint img_row	= base_row + margin;
	//uint img_col	= base_col + margin;
	//uint img_index	= img_row*(cols + 2*margin) + img_col;   									// NB here use cols not mm_cols
	
	uint read_index = read_offset_  +  base_row  * mm_cols  + base_col  ;							// NB 4 channels.  + margin

	float4 temp_float4  = {H/(2*M_PI_F),S,V,1.0f};													// Note how to load a float4 vector. Also H->(0,1) for display.
	if (global_id <= pixels) {
		img[read_index] 		= temp_float4;
		local_sum_pix[lid] 		= temp_float4;
	}else local_sum_pix[lid]	= 0;
	
	/* from https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
	 * In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and scaled to fit the 0 to 1 range.
	 * V = max(R,G,B)
	 * S = (0, if V=0), otherwise (V-min(RGB))/V
	 * H = 60(G-B)/(V-min(R,G,B)  			if V=R
	 *     120 + 60(B-R)/(V-min(R,G,B))		if V=G
	 *     240 + 60(R-G)/(V-min(R,G,B))		if V=B
	 *     0								if R=G=B
	 */
	///////////////////////////////////////////////////////////										// Sum pixels in the work group, using local mem.
	
	int max_iter = ilogb((float)(group_size));
	
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?  
																									// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size)  local_sum_pix[lid] += local_sum_pix[lid+group_size];					// local_sum_pix  
	}
	if (lid==0) {
		uint group_id 			= get_group_id(0);
		uint global_sum_offset 	= 0; //read_offset_ / local_size ;		// only the base layer		// Compute offset for this layer
		uint num_groups 		= get_num_groups(0);
		/*
		printf("\n__kernel cvt_color_space_linear(..) reduction=%u,  global_sum_offset=%u,  num_groups=%u,  group_id=%u,     local_sum_pix[lid]=( %f, %f, %f, %f),    global_sum_pix[group_id]=( %f, %f, %f, %f ) "\
		, reduction, global_sum_offset,  num_groups, group_id \
		, local_sum_pix[lid][0],local_sum_pix[lid][1],local_sum_pix[lid][2],local_sum_pix[lid][3] \
		, global_sum_pix[group_id][0], global_sum_pix[group_id][1], global_sum_pix[group_id][2], global_sum_pix[group_id][3] \
		);
		*/
		float4 layer_data = {num_groups, reduction, 0.0f, 0.0f };			// Write layer data to first entry
		if (global_id == 0) {global_sum_pix[global_sum_offset] = layer_data; }
		global_sum_offset += 1+ group_id;
		
		if (local_sum_pix[0][3] >0){																// Using alpha channel local_sum_pix[0][3], to count valid pixels being summed.
			global_sum_pix[global_sum_offset] = local_sum_pix[0] / local_sum_pix[0][3];				// Save to global_sum_pix // Count hits, and divide group by num hits, without using atomics!
		}else global_sum_pix[global_sum_offset] = 0;
	}
}




__kernel void image_variance(
	__global	float4*	img_stats,		//0
	__global	float4*	img,			//1
	__constant	uint*	uint_params,	//2
	__constant 	uint8*	mipmap_params,	//3
	__local		float4*	local_sum_var,	//4
	__global	float4*	global_sum_var	//5
			  )
{
	int global_id 	= (int)get_global_id(0);
	uint pixels 	= uint_params[PIXELS];
	if (global_id > pixels) return;
	uint lid 				= get_local_id(0);
	uint local_size 		= get_local_size(0);
	uint group_size 		= local_size;
	uint reduction			= 1;																	// = mm_cols/read_cols_; but this is only the baselayer ofthe image pyramid.
	
	uint8 mipmap_params_ = mipmap_params[0];
	uint read_offset_ 	 = mipmap_params_[MiM_READ_OFFSET];
	// 
	uint cols 		= uint_params[COLS];
	uint margin 	= uint_params[MARGIN];
	uint mm_cols	= uint_params[MM_COLS];
	
	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	uint img_row	= base_row + margin;
	uint img_col	= base_col + margin;
	uint img_index	= img_row*(cols + 2*margin) + img_col;   										// NB here use cols not mm_cols
	
	uint read_index = read_offset_  +  base_row  * mm_cols  + base_col  ;							// NB 4 channels.  + margin
	
	float4 variance = powr( (img[read_index]-img_stats[IMG_MEAN]), 2);								// TODO why does this cause NaNs ?
	
	int4 var_isnan = isnan(variance);
	if (global_id <= pixels && !var_isnan.x && !var_isnan.y && !var_isnan.z) {
		local_sum_var[lid] 		= variance;
	}else local_sum_var[lid]	= 0;
	/*
	if (global_id==0) printf("\n__kernel image_variance(..) 1, read_index=%u,   img_stats[IMG_MEAN]=%f,  img[read_index]=(%f,%f,%f,%f),  variance=(%f,%f,%f,%f)"\
							  , read_index,      img_stats[IMG_MEAN], img[read_index].x, img[read_index].y, img[read_index].z, img[read_index].w,  variance.x, variance.y, variance.z, variance.w \
					  );
	*/
	///////////////////////// reduction
	
	int max_iter = ilogb((float)(group_size));
	
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?  
																									// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size)  local_sum_var[lid] += local_sum_var[lid+group_size];					// local_sum_var  
	}
	if (lid==0) {
		uint group_id 			= get_group_id(0);
		uint global_sum_offset 	= 0; //read_offset_ / local_size ;		// only the base layer								// Compute offset for this layer
		uint num_groups 		= get_num_groups(0);
		/*
		printf("\n__kernel image_variance(..) 2, reduction=%u,  global_sum_offset=%u,  num_groups=%u,  group_id=%u, local_sum_var[lid]=( %f, %f, %f, %f),  global_sum_var[group_id]=( %f, %f, %f, %f ) "\
		, reduction, global_sum_offset,  num_groups, group_id \
		, local_sum_var[lid][0],local_sum_var[lid][1],local_sum_var[lid][2],local_sum_var[lid][3] \
		, global_sum_var[group_id][0], global_sum_var[group_id][1], global_sum_var[group_id][2], global_sum_var[group_id][3] \
		);
		*/
		float4 layer_data = {num_groups, reduction, 0.0f, 0.0f };			// Write layer data to first entry
		if (global_id == 0) {global_sum_var[global_sum_offset] = layer_data; }
		global_sum_offset += 1+ group_id;
		
		if (local_sum_var[0][3] >0){																// Using alpha channel local_sum_var[0][3], to count valid pixels being summed.
			global_sum_var[global_sum_offset] = local_sum_var[0] / local_sum_var[0][3];				// Save to global_sum_var // Count hits, and divide group by num hits, without using atomics!
		}else global_sum_var[global_sum_offset] = 0;
	}
}

__kernel void blur_image(
	__private	uint	layer,			//0															// Intitial lyer bluring needs a different buffer to write blurred image to. 
	__constant 	uint8*	mipmap_params,	//1															// => the original loading should be to an interim buffer. 
	__constant 	float* 	gaussian,		//2
	__constant 	uint*	uint_params,	//3
	__global 	float4*	img,			//4
	__global	float4* img_blurred,	//5
	__local	 	float4*	local_img_patch //6
			  )
{
	int global_id_u 		= (int)get_global_id(0);
	float global_id_flt 	= global_id_u;
	uint lid 				= get_local_id(0);
	uint group_size 		= get_local_size(0);
	uint patch_length		= group_size+4;
	uint pixels 			= uint_params[PIXELS];
	
	uint8 mipmap_params_	= mipmap_params[0];
	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
	uint cols 				= uint_params[COLS];
	uint mm_cols			= uint_params[MM_COLS];
	
	uint base_row			= global_id_u/cols ;
	uint base_col			= global_id_u%cols ;
	uint read_index 		= read_offset_  +  base_row  * mm_cols  + base_col  ;
	
	uint read_rows_			= mipmap_params_[MiM_READ_ROWS];
	uint read_cols_ 		= mipmap_params_[MiM_READ_COLS];
	uint read_row   		= global_id_u / read_cols_ ;
	uint read_column		= fmod(global_id_flt, read_cols_);
	
	for (int i=0, j=-2; i<5; i++, j++){																// Load local_img_patch
		local_img_patch[lid+2 + i*patch_length] = img[ read_index +j*mm_cols];
	}
	////
	if (lid==0 || lid==1){
		for (int i=0; i<5; i++){
			local_img_patch[lid + i*patch_length] = img[ read_index +i*mm_cols -2]; //white; //
		}
	}
	if (lid==group_size-2 || lid==group_size-1){
		for (int i=0; i<5; i++){
			local_img_patch[lid+4 + i*patch_length] = img[ read_index +i*mm_cols +2]; //black; //
		}
	}
	////
	if ((read_row>read_rows_-3) ||  (read_row < 3)  ){												// Prevents blurring with black space below the image.
		for (int i=0; i<5; i++){
			local_img_patch[lid+2 + i*patch_length] = img[ read_index ];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);																	// No 'if->return' before fence between write & read local mem
	float4 blurred_pixel = 0;
	for (int i=0; i<5; i++){
		for (int j=0; j<5; j++){
			blurred_pixel += local_img_patch[lid+j + i*patch_length]/25; 							// 5x5 box filter, rather than Gaussian
		}
	}
	if (read_column < 2 || read_column > read_cols_ -3) {
		blurred_pixel = 0;
		for (int i=0; i<5; i++){
			blurred_pixel += local_img_patch[lid+2 + i*patch_length]/5;								// prevents blur wrapping left-right.
		}
	}
	if (read_row>=read_rows_ || global_id_u >= pixels) return;									// num pixels to be written & num threads to really use. // mipmap_params_[MiM_PIXELS]
	
	blurred_pixel[3] = 1.0f;
	img_blurred[ read_index] = blurred_pixel;
}


__kernel void mipmap_linear_flt4(																	// Nvidia Geforce GPUs cannot use "half"
	__private	uint	layer,			//0
	__constant 	uint8*	mipmap_params,	//1
	__constant 	float* 	gaussian,		//2
	__constant 	uint*	uint_params,	//3
	__global 	float4*	img,			//4
	__local	 	float4*	local_img_patch //5
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	uint patch_length	= group_size+4;
																	if (global_id_u ==0){printf("\n\n__kernel void mipmap_linear_flt(..), __private	uint	layer=%u", layer);}
	uint8 mipmap_params_ = mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint write_offset_ 	= mipmap_params_[MiM_WRITE_OFFSET]; 										// = read_offset_ + read_cols_*read_rows for linear MipMap.
	uint read_rows_		= mipmap_params_[MiM_READ_ROWS];
	uint write_rows_	= read_rows_ /2;
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint write_cols_ 	= mipmap_params_[MiM_WRITE_COLS];
	
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];   													// whole mipmap
	
	uint write_row   	= global_id_u / write_cols_ ;
	uint write_column 	= fmod(global_id_flt, write_cols_);
	
	uint read_row    	= 2*write_row;
	uint read_column 	= 2*write_column;
	
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column  ;					// NB 4 channels.  + margin
	uint write_index 	= write_offset_ +  write_row * mm_cols  + write_column ;					// write_cols_, use read_cols_ as multiplier to preserve images  + margin
	
	//float4 white = {1.0f,1.0f,1.0f,1.0f};
	//float4 black = {0.0f,0.0f,0.0f,0.0f};
	
	for (int i=0, j=-2; i<5; i++, j++){																// Load local_img_patch
		local_img_patch[lid+2 + i*patch_length] = img[ read_index +j*mm_cols];
	}
	////
	if (lid==0 || lid==1){
		for (int i=0; i<5; i++){
			local_img_patch[lid + i*patch_length] = img[ read_index +i*mm_cols -2]; //white; //
		}
	}
	if (lid==group_size-2 || lid==group_size-1){
		for (int i=0; i<5; i++){
			local_img_patch[lid+4 + i*patch_length] = img[ read_index +i*mm_cols +2]; //black; //
		}
	}
	////
	if ((write_row>write_rows_-3) ||  (write_row < 3)  ){											// Prevents blurring with black space below the image.
		for (int i=0; i<5; i++){
			local_img_patch[lid+2 + i*patch_length] = img[ read_index ];
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);																	// No 'if->return' before fence between write & read local mem
	float4 reduced_pixel = 0;
	for (int i=0; i<5; i++){
		for (int j=0; j<5; j++){
			reduced_pixel += local_img_patch[lid+j + i*patch_length]/25; 							// 5x5 box filter, rather than Gaussian
		}
	}
	
	if (write_column < 2 || write_column > write_cols_ -3) {
		reduced_pixel = 0;
		for (int i=0; i<5; i++){
			reduced_pixel += local_img_patch[lid+2 + i*patch_length]/5;								// prevents blur wrapping left-right.
		}
	}
	
	if (write_row>=write_rows_) return;
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;											// num pixels to be written & num threads to really use.
	/*
	if (global_id_u == 1) printf("\n\npatch_length=%u,  group_size=%u, mm_cols=%u, mipmap_params_[MiM_PIXELS]=%u", patch_length, group_size, mm_cols, mipmap_params_[MiM_PIXELS] );
	if (global_id_u < 5) printf("\n\nread_index=%u, write_index=%u, lid=%u, write_cols_=%u,    read_index+0*mm_cols=%u, read_index+1*mm_cols=%u, read_index+2*mm_cols=%u, read_index+3*mm_cols=%u, read_index+4*mm_cols=%u,       "\
		, read_index, write_index, lid, write_cols_ \
		, read_index+0*mm_cols, read_index+1*mm_cols, read_index+2*mm_cols, read_index+3*mm_cols, read_index+4*mm_cols \
	);
	*/
	//reduced_pixel[2] = global_id_flt/(float)(mipmap_params[MiM_PIXELS]); // debugging 
	reduced_pixel[3] = 1.0f;
	img[ write_index] = reduced_pixel;
}

__kernel void mipmap_linear_flt(																	// Nvidia Geforce GPUs cannot use "half"
	__private	uint	layer,			//0
	__constant 	uint8*	mipmap_params,	//1
	__constant 	float* 	gaussian,		//2
	__constant 	uint*	uint_params,	//3
	__global 	float*	img,			//4
	__local	 	float*	local_img_patch	//5
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	uint patch_length	= group_size+4;
	/*
																	if (global_id_u ==0){printf("\n\n__kernel void mipmap_linear_flt(..), __private	uint	layer=%u", layer);}
	*/
	uint8 mipmap_params_ = mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint write_offset_ 	= mipmap_params_[MiM_WRITE_OFFSET]; 										// = read_offset_ + read_cols_*read_rows for linear MipMap.
	uint read_rows_		= mipmap_params_[MiM_READ_ROWS];
	uint write_rows_	= read_rows_ /2;
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint write_cols_ 	= mipmap_params_[MiM_WRITE_COLS];
	
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];   													// whole mipmap
	
	uint write_row   	= global_id_u / write_cols_ ;
	uint write_column 	= fmod(global_id_flt, write_cols_);
	
	uint read_row    	= 2*write_row;
	uint read_column 	= 2*write_column;
	
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column  ;					// NB 4 channels.  + margin
	uint write_index 	= write_offset_ +  write_row * mm_cols  + write_column ;					// write_cols_, use read_cols_ as multiplier to preserve images  + margin
	
	for (int i=0, j=-2; i<5; i++, j++){																// Load local_img_patch
		local_img_patch[lid+2 + i*patch_length] = img[ read_index +j*mm_cols];
	}
	////
	if (lid==0 || lid==1){
		for (int i=0; i<5; i++){
			local_img_patch[lid + i*patch_length] = img[ read_index +i*mm_cols -2]; //white; //
		}
	}
	if (lid==group_size-2 || lid==group_size-1){
		for (int i=0; i<5; i++){
			local_img_patch[lid+4 + i*patch_length] = img[ read_index +i*mm_cols +2]; //black; //
		}
	}
	////
	if ((write_row>write_rows_-3) ||  (write_row < 3)  ){											// Prevents blurring with black space below the image.
		for (int i=0; i<5; i++){
			local_img_patch[lid+2 + i*patch_length] = img[ read_index ];
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);																	// No 'if->return' before fence between write & read local mem
	float reduced_pixel = 0;
	for (int i=0; i<5; i++){
		for (int j=0; j<5; j++){
			reduced_pixel += local_img_patch[lid+j + i*patch_length]/25; 							// 5x5 box filter, rather than Gaussian
		}
	}
	
	if (write_column < 2 || write_column > write_cols_ -3) {
		reduced_pixel = 0;
		for (int i=0; i<5; i++){
			reduced_pixel += local_img_patch[lid+2 + i*patch_length]/5;								// prevents blur wrapping left-right.
		}
	}
	
	if (write_row>=write_rows_) return;
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;											// num pixels to be written & num threads to really use.
	
	img[ write_index] = reduced_pixel;
}

__kernel void  img_grad(
	__private	uint		layer,			//0
	__constant	uint8*		mipmap_params,	//1
	__constant 	uint*		uint_params,	//2
	__constant 	float*		fp32_params,	//3
	__global 	float4*		img,			//4 
	__global 	float4*		gxp,			//5
	__global 	float4*		gyp,			//6
	__global 	float4*		g1p,			//7
	__constant 	float2*		SE3_map,		//8
	//__global	float* 		depth_map,		//9														// NB GT_depth, not inv_depth
	__global 	float8*		SE3_grad_map,	//9														// We keep hsv sepate at this stage, so 6*4*2=24, but float16 is the largest type, so 6*float8.
	__global 	float8*		HSV_grad		//10	
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	
	uint8 mipmap_params_ = mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	
	uint base_cols		= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint mm_pixels		= uint_params[MM_PIXELS];
	
	uint read_row    	= global_id_u / read_cols_;
	uint read_column 	= fmod(global_id_flt, read_cols_);
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column ;						// NB 4 channels.  + margin
	
	/// adapted
	int upoff			= -(read_row  >1 )*mm_cols;													//-(read_row  != 0)*mm_cols;				// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;
    int lfoff			= -(read_column >1);														//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);											// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;
	/*
	if (read_row<8 && read_column == 15)   										printf("\nreadrow=%u, -(read_row >1 )=%i, upoff=%i ",read_row, -(read_row >1), upoff );
	if (read_row<=read_rows_ && read_row>=read_rows_-4 && read_column == 15)	printf("\nreadrow=%u, (read_row < read_rows_-2)=%i, dnoff=%i ",read_row, (read_row < read_rows_-2), dnoff );
	
	if (read_column <8 && read_row== 15)   												printf("\nread_column=%u, -(read_column >1)=%i, lfoff=%i ",read_column, -(read_column >1), lfoff );
	if (read_column<read_cols_ && read_column>=read_cols_-8 && read_row == 15)	printf("\nread_column=%u, (read_column < read_cols_-2)=%i, rtoff=%i ",read_column, (read_column < read_cols_-2), rtoff );
	*/
	
	float alphaG		= fp32_params[ALPHA_G];														// "alpha_g":0.015,//15,	// can vary 15 to 0.15  // 0.015     
																									// DTAM paper beta for optimization, (not for g1 edges): beta=0.001 while theta>0.001, else beta=0.0001 
	float betaG 		= fp32_params[BETA_G];														// "beta_g": 1.5,			//my dtam_opencl 1.5 
	
	float4 pu, pd, pl, pr;
	pr =  img[offset + rtoff];
	pl =  img[offset + lfoff];
	pu =  img[offset + upoff];
	pd =  img[offset + dnoff];

	float4 gx	= { (pr.x - pl.x), (pr.y - pl.y), (pr.z - pl.z), 1.0f };							// Signed img gradient in hsv
	float4 gy	= { (pd.x - pu.x), (pd.y - pu.y), (pd.z - pu.z), 1.0f };
	 
	float4 g1  = { \
		 exp(-alphaG * pow(sqrt(gx.x*gx.x + gy.x*gy.x), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.y*gx.y + gy.y*gy.y), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.z*gx.z + gy.z*gy.z), betaG) ), \
		 1.0f };																					// exp(-0.015  * pow(sqrt(gx.x*gx.x + gy.x*gy.x),  1.5)  )
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;	 
	g1p[offset]= g1;
	gxp[offset]= fabs(gx);																			// NB taking the absolute loses the sign of the gradient. 
	gyp[offset]= fabs(gy);
	
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;													// read_row
	uint u 				= fmod(global_id_flt, read_cols_);											// read_column
	//uint depth_index	= v * reduction * base_cols + u * reduction;								// Sparse sampling of the depth map of the base image.
	/*
	float depth = depth_map[depth_index] ;
	float inv_depth = 0.0f;						
	if (depth!=0){ inv_depth = 80/depth; }															// inv_depth   range 1/255 to 1		// NB ahanda daaset depth range 0-255.
	float rotation_wt = 1 - inv_depth;																// rotation_wt range 1 to 1/255		furthest to closest
	*/
	for (uint i=0; i<3; i++) {	
		float2 SE3_px =  SE3_map[read_index + i* mm_pixels];										// SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;   float2 partial_gradient={u_flt-u2 , v_flt-v2}; // Find movement of pixel
		float8 SE3_grad_px = {gx*SE3_px[0]  , gy*SE3_px[1] };										//  NB float4 gx, gy => float8
		SE3_grad_map[read_index + i* mm_pixels]  =  SE3_grad_px ;//* rotation_wt; 
		/*
		if (global_id_u ==1)printf("\nSE3_grad_map[%u + %u * %u] = ((%f, %f, %f, %f),  (%f, %f, %f, %f)),  SE3_px=(%f,%f),  gx=(%f, %f, %f, %f),  gy=(%f, %f, %f, %f)" \
		, read_index, i, mm_pixels,   SE3_grad_px[0], SE3_grad_px[1], SE3_grad_px[2], SE3_grad_px[3],   SE3_grad_px[4], SE3_grad_px[5], SE3_grad_px[6], SE3_grad_px[7]\
		,    SE3_px[0], SE3_px[1]  \
		, gx[0], gx[1], gx[2], gx[3],   gy[0], gy[1], gy[2], gy[3] \
				  );
		*/
	} 
	for (uint i=3; i<6; i++) {	
		float2 SE3_px =  SE3_map[read_index + i* mm_pixels];										// SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;   float2 partial_gradient={u_flt-u2 , v_flt-v2}; // Find movement of pixel
		float8 SE3_grad_px = {gx*SE3_px[0]  , gy*SE3_px[1] };										//  NB float4 gx, gy => float8
		SE3_grad_map[read_index + i* mm_pixels]  =  SE3_grad_px ;//* inv_depth ; 
		/*
		if (global_id_u ==1) printf("\nSE3_grad_map[%u + (%u * %u)] = ((%f, %f, %f, %f),  (%f, %f, %f, %f)),  SE3_px=(%f,%f),  gx=(%f, %f, %f, %f),  gy=(%f, %f, %f, %f)" \
		, read_index, i, mm_pixels,   SE3_grad_px[0], SE3_grad_px[1], SE3_grad_px[2], SE3_grad_px[3],   SE3_grad_px[4], SE3_grad_px[5], SE3_grad_px[6], SE3_grad_px[7]\
		,    SE3_px[0], SE3_px[1]  \
		, gx[0], gx[1], gx[2], gx[3],   gy[0], gy[1], gy[2], gy[3] \
				  );
		*/
	}
	
	float H = img[offset][0] * 2*M_PI_F;
	float S = img[offset][1];
	float V = img[offset][2];
	float8 temp_float8  = { sin(H) , cos(H), S, V, gx[1], gy[1], gx[2], gy[2] }; // 0.0f, 0.0f }; //  gx[2], gy[2] };   pr.z, 0.5 // 0.0f, 0.0f,   sin(H)
	
	HSV_grad[offset] = temp_float8;
}

__kernel void compute_param_maps(
	__private	uint	layer,			//0
	__constant 	uint8*	mipmap_params,	//1
	__constant 	uint*	uint_params,	//2
	__constant 	float* 	SO3_k2k,		//3
	__global 	float2*	SE3_map			//4
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	
	uint8 mipmap_params_ = mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	/*
																	if (global_id_u ==0) {printf("\n__kernel void mipmap_linear_flt(..), \nlayer=%u,  \nread_offset_=%u,  \nread_cols_=%u,  \nmipmap_params_[MiM_PIXELS]=%u, \nmipmap_params[layer][0]=%u ",\
																		layer,  read_offset_,  read_cols_, mipmap_params_[MiM_PIXELS], mipmap_params[layer][0] );}
	*/
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;
	
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;													// read_row
	uint u 				= fmod(global_id_flt, read_cols_);											// read_column
	float u_flt			= u * reduction;															// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	/*
	//if (global_id_u == 0 || global_id_u == mipmap_params[MiM_PIXELS]-1 ) printf("\nreduction=%u,  read_offset_=%u  , read_cols_=%u,  mipmap_params[MiM_PIXELS]=%u, global_id_u=%u,   u=%u,  v=%u, u_flt=%f, v_flt=%f,  uint_params[MM_PIXELS]=%u,  ", \
	//	reduction, mipmap_params[MiM_READ_OFFSET], mipmap_params[MiM_READ_COLS], mipmap_params[MiM_PIXELS], global_id_u, u, v, u_flt, v_flt, uint_params[MM_PIXELS]  );
	*/
	for (uint i=0; i<6; i++) {																		// for each SE3 DoF
																									// Find new pixel position, h=homogeneous coords.
		int idx = i *16;
		float inv_depth = 1.0f;																		// mid point max-min inv depth
		float uh2 = SO3_k2k[idx+0]*u_flt + SO3_k2k[idx+1]*v_flt + SO3_k2k[idx+2]*1 + SO3_k2k[idx+3]*inv_depth;
		float vh2 = SO3_k2k[idx+4]*u_flt + SO3_k2k[idx+5]*v_flt + SO3_k2k[idx+6]*1 + SO3_k2k[idx+7]*inv_depth;
		float wh2 = SO3_k2k[idx+8]*u_flt + SO3_k2k[idx+9]*v_flt + SO3_k2k[idx+10]*1+ SO3_k2k[idx+11]*inv_depth;
		//float h/z  = SO3_k2k[12]*u_flt + SO3_k2k[13]*v + SO3_k2k[14]*1; 							// +SO3_k2k[15]/z
	
		float u2   = uh2/wh2;
		float v2   = vh2/wh2;
		float2 partial_gradient={u_flt-u2 , v_flt-v2}; 												// Find movement of pixel
		/*
		// (global_id_u == 10 || global_id_u == 11)
		//if  (u==0 && v <5) printf("\n partial_gradient= { %f - %f = %f,  %f - %f = %f }, i=%u,     uint_params[MM_PIXELS]=%u,    (read_index + i* uint_params[MM_PIXELS])=%u     ",  \
		//	 u_flt,  u2, (u_flt-u2), v_flt, v2, (v_flt-v2), i, uint_params[MM_PIXELS], read_index+i*uint_params[MM_PIXELS]  );
		*/
		SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;
		//if (lid==0){printf("\ni=%u partial_gradient[0]=%f, partial_gradient[1]=%f",i, partial_gradient[0], partial_gradient[1]);}
	}
	
	// TODO // Create a 'reproject' & 'img_grad_sum' kernels 
}

/*
 * For Intel iRIS Xe 
// Max number of constant args                     8
// Max constant buffer size                        4294959104 (4GiB)
NB shoud use these for things that never change during runtime, not for variables constant in a particular kernel but not another.
TODO Declare constants at top of the device prgram file.
*/

__kernel void so3_grad(
	__private	uint	layer,					//0
	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3
	__global	float*  so3_k2k,				//4
	__global 	float4*	img_cur,				//5 
	__global 	float4*	img_new,				//6
	__global 	float8*	SE3_grad_map_cur_frame,	//7
	__global 	float8*	SE3_grad_map_new_frame,	//8
	//__global	float* 	depth_map,				//9													// NB GT_depth, not inv_depth
	__local		float4*	local_sum_grads,		//10
	__global	float4*	global_sum_grads,		//11
	__global 	float4*	SE3_incr_map_,			//12
	__global	float4* Rho_,					//13
	__local		float4*	local_sum_rho_sq,		//14												// 1 DoF, float4 channels
	__global 	float4*	global_sum_rho_sq		//15
	)
 {																									// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	
	uint local_size 	= get_local_size(0);
	uint group_size 	= local_size;
	uint work_dim 		= get_work_dim();
	uint global_size	= get_global_size(0);
	
	uint8 mipmap_params_= mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels	= mipmap_params_[MiM_PIXELS];
	
	uint base_cols		= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint mm_pixels		= uint_params[MM_PIXELS];
	
	float SE3_LM_a		= fp32_params[SE3_LM_A];													// Optimisation parameters
	float SE3_LM_b		= fp32_params[SE3_LM_B];
	
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;													// read_row
	uint u 				= fmod(global_id_flt, read_cols_);											// read_column
	float u_flt			= u * reduction;															// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	float alpha			= img_cur[read_index].w;
	
	float uh2 = so3_k2k[0]*u_flt + so3_k2k[1]*v_flt + so3_k2k[2]*1;
	float vh2 = so3_k2k[4]*u_flt + so3_k2k[5]*v_flt + so3_k2k[6]*1;
	float wh2 = so3_k2k[8]*u_flt + so3_k2k[9]*v_flt + so3_k2k[10]*1;
	
	float u2_flt	= uh2/wh2;
	float v2_flt	= vh2/wh2;
	int  u2			= floor((u2_flt/reduction)+0.5f) ;												// nearest neighbour interpolation
	int  v2			= floor((v2_flt/reduction)+0.5f) ;												// NB this corrects the sparse sampling to the redued scales.
	
	uint read_index_new = read_offset_ + v2 * mm_cols  + u2; 										// read_cols_
	uint num_DoFs = 3;
	/*
	if (global_id_u==1){
		printf("\nkernel so3_grad(..)  so3_k2k=( %f,%f,%f,%f   ,%f,%f,%f,%f    ,%f,%f,%f,%f  ,%f,%f,%f,%f)"\
		,so3_k2k[0],so3_k2k[1],so3_k2k[2],so3_k2k[3],  so3_k2k[4],so3_k2k[5],so3_k2k[6],so3_k2k[7],  so3_k2k[8],so3_k2k[9],so3_k2k[10],so3_k2k[11],  so3_k2k[12],so3_k2k[13],so3_k2k[14],so3_k2k[15] );
	}
	*/
	float4 rho = {0.0f,0.0f,0.0f,0.0f}, zero_f4={0.0f,0.0f,0.0f,0.0f}; 
	float intersection = (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2>2) && (u2<=read_cols_-2) && (v2>2) && (v2<=read_rows_-2)  &&  (global_id_u<=layer_pixels);
	if (!intersection) read_index_new = read_index;
		
	for (int i=0; i<num_DoFs; i++) local_sum_grads[i*local_size + lid] = 0;							// Essential to zero local mem.
	if (  intersection  ) {																			// if (not cleanly within new frame) skip  Problem u2&v2 are wrong.
		int idx = 0;
		rho     = img_cur[read_index] - img_new[read_index_new];
		rho[3]  = alpha;
	}
	Rho_[read_index]      = rho;																	// save pixelwise photometric error map to buffer. NB Outside if(){}, to zero non-overlapping pixels.
	float4 rho_sq         = {rho.x*rho.x,  rho.y*rho.y,  rho.z*rho.z, rho.w};
	local_sum_rho_sq[lid] = rho_sq;																	// Also compute global Rho^2.
	
	/*
	if (u<10 && v==10){
			printf("\nreduction=%u,  global_id_u=%u, u=%u, u_flt=%f,  uh2=%f, wh2=%f, u2_flt=%f, u2=%u,  rho=(%f,%f,%f,%f),        intersection=%f "\
					 ,reduction,     global_id_u,    u,    u_flt,     uh2,    wh2,    u2_flt,    u2,     rho.x,rho.y,rho.z,rho.w,  intersection);
	}
	*/
	//if (intersection >0 ) {printf("\n  rho=(%f,%f,%f,%f)", rho.x,rho.y,rho.z,rho.w );}
	
	for (uint i=0; i<num_DoFs; i++) {																// for each SO3 DoF
		float4 delta4 = zero_f4;
		//if ( intersection ){ 																		// never read outside the buffer.
			float8 SE3_grad_cur_px = SE3_grad_map_cur_frame[read_index     + i * mm_pixels ] ;
			float8 SE3_grad_new_px = SE3_grad_map_new_frame[read_index_new + i * mm_pixels ] ;
			delta4.w=alpha;
			for (int j=0; j<3; j++)  delta4[j] = rho[j] * (SE3_grad_cur_px[j] + SE3_grad_cur_px[j+4] + SE3_grad_new_px[j] + SE3_grad_new_px[j+4]);
		//}
		local_sum_grads[i*local_size + lid] = delta4;												// write grads to local mem for summing over the work group.
		SE3_incr_map_[read_index + i * mm_pixels ] = delta4;
	}
	for (uint i=3; i<6; i++)SE3_incr_map_[read_index + i * mm_pixels ]=zero_f4;						// zero the unused DoFs.
	
	////////////////////////////////////////////////////////////////////////////////////////		// Reduction
	
	int max_iter = 9;//ceil(log2((float)(group_size)));
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?  NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size){
			for (int i=0; i<num_DoFs; i++){
				local_sum_grads[i*local_size + lid] += local_sum_grads[i*local_size + lid + group_size];	// local_sum_grads  
			}
			local_sum_rho_sq[lid] += local_sum_rho_sq[lid + group_size];							// Also compute global Rho^2.
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid==0) {
		uint group_id 				= get_group_id(0);
		uint rho_global_sum_offset 	= read_offset_ / local_size ;									// Compute offset for this layer
		uint se3_global_sum_offset 	= rho_global_sum_offset *num_DoFs;								// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
		uint num_groups 			= get_num_groups(0);
		/*
		printf("\n__kernel se3_grad(..) reduction=%u,  se3_global_sum_offset=%u,  num_groups=%u,  group_id=%u, read_index=%u,  local_sum_rho_sq[lid]=(%f,%f,%f,%f)     local_sum_grads[lid]=", reduction, se3_global_sum_offset,  num_groups, group_id, read_index, local_sum_rho_sq[lid].x, local_sum_rho_sq[lid].y, local_sum_rho_sq[lid].z, local_sum_rho_sq[lid].w );
		for (int i=0; i<num_DoFs; i++){
			printf("\n  group_id=%u, i=%d, (%f, %f, %f, %f),  max_iter=%d",  group_id, i, local_sum_grads[i*local_size+lid].x, local_sum_grads[i*local_size+lid].y, local_sum_grads[i*local_size+lid].z, local_sum_grads[i*local_size+lid].w,  max_iter );
		}
		*/
		
		float4 layer_data = {num_groups, reduction, 0.0f, 0.0f };									// Write layer data to first entry
		if (global_id_u == 0) {
			global_sum_rho_sq[rho_global_sum_offset] 	= layer_data;
			global_sum_grads[se3_global_sum_offset] 	= layer_data; 
		}
		rho_global_sum_offset += 1 + group_id;
		se3_global_sum_offset += num_DoFs+ group_id*num_DoFs;
		
		if (local_sum_grads[0][3] >0){																// Using last channel local_sum_pix[0][7], to count valid pixels being summed.
			global_sum_rho_sq[rho_global_sum_offset] = local_sum_rho_sq[lid];
			for (int i=0; i<num_DoFs; i++){
				float4 temp_float4 = local_sum_grads[i*local_size + lid] / local_sum_grads[i*local_size + lid].w ;
				global_sum_grads[se3_global_sum_offset + i] = temp_float4 ;							// local_sum_grads  
			}																						// Save to global_sum_grads // Count hits, and divide group by num hits, without using atomics!
		}else {																						// If no matching pixels in this group, set values to zero.
			global_sum_rho_sq[rho_global_sum_offset] = 0;
			for (int i=0; i<num_DoFs; i++){
				global_sum_grads[rho_global_sum_offset] = 0;
			}
		}
		
		/*
		printf("\n global_sum_grads[%u] =  (%f,%f,%f,%f),  local_sum_rho_sq[%u] = (%f,%f,%f,%f)"\
		, rho_global_sum_offset, global_sum_grads[rho_global_sum_offset].x, global_sum_grads[rho_global_sum_offset].y, global_sum_grads[rho_global_sum_offset].z, global_sum_grads[rho_global_sum_offset].w \
		, lid,  local_sum_rho_sq[lid].x,  local_sum_rho_sq[lid].y,  local_sum_rho_sq[lid].z,  local_sum_rho_sq[lid].w  );
		*/
	}
}

__kernel void se3_grad(
	__private	uint	layer,					//0
	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3
	__global	float16*k2k,					//4		// keyframe2K
	__global 	float4*	img_cur,				//5		// keyframe	
	__global 	float4*	img_new,				//6
	__global 	float8*	SE3_grad_map_cur_frame,	//7		// keyframe
	__global 	float8*	SE3_grad_map_new_frame,	//8
	__global	float* 	depth_map,				//9		// NB keyframe GT_depth, now stored as inv_depth
	__local		float4*	local_sum_grads,		//10
	__global	float4*	global_sum_grads,		//11
	__global 	float4*	SE3_incr_map_,			//12
	__global	float4* Rho_,					//13
	__local		float4*	local_sum_rho_sq,		//14	1 DoF, float4 channels
	__global 	float4*	global_sum_rho_sq		//15
	)
 {																									// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	
	uint local_size 	= get_local_size(0);
	uint group_size 	= local_size;
	uint work_dim 		= get_work_dim();
	uint global_size	= get_global_size(0);
	float16 k2k_pvt		= k2k[0];
	
	uint8 mipmap_params_ = mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels	= mipmap_params_[MiM_PIXELS];
	
	uint base_cols		= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint mm_pixels		= uint_params[MM_PIXELS];
	
	float SE3_LM_a		= fp32_params[SE3_LM_A];													// Optimisation parameters
	float SE3_LM_b		= fp32_params[SE3_LM_B];
	
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;													// read_row
	uint u 				= fmod(global_id_flt, read_cols_);											// read_column
	float u_flt			= u * reduction;															// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	float alpha			= img_cur[read_index].w;
	
	float inv_depth 	= depth_map[read_index /*depth_index*/]; 									//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.//inv dept
	float uh2 = k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1 + k2k_pvt[3]*inv_depth;
	float vh2 = k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1 + k2k_pvt[7]*inv_depth;
	float wh2 = k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1+ k2k_pvt[11]*inv_depth;
	//float h/z  = k2k_pvt[12]*u_flt + k2k_pvt[13]*v + k2k_pvt[14]*1; // +k2k_pvt[15]/z
	
	float u2_flt	= uh2/wh2;
	float v2_flt	= vh2/wh2;
	int  u2			= floor((u2_flt/reduction)+0.5f) ;												// nearest neighbour interpolation
	int  v2			= floor((v2_flt/reduction)+0.5f) ;												// NB this corrects the sparse sampling to the redued scales.
	uint read_index_new = read_offset_ + v2 * mm_cols  + u2; // read_cols_
	uint num_DoFs 	= 6;
	
	if (global_id_u==1){
		printf("\nkernel se3_grad(..): u=%i,  v=%i,   inv_depth=%f, u2=%f,  v2=%f,  int_u2=%i,  int_v2=%i,    k2k_pvt=(%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f)"\
		,u, v, inv_depth, u_flt, v_flt, u2, v2,  k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],   k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],   k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15]   )  ;
	}
	
	float4 rho = {0.0f,0.0f,0.0f,0.0f}; 															// TODO apply robustifying norm to Rho, eg Huber norm.
	float intersection = (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2>2) && (u2<=read_cols_-2) && (v2>2) && (v2<=read_rows_-2)  &&  (global_id_u<=layer_pixels);
	
	for (int i=0; i<6; i++) local_sum_grads[i*local_size + lid] = 0;								// Essential to zero local mem.
	if (  intersection  ) {																			// if (not cleanly within new frame) skip  Problem u2&v2 are wrong.
		int idx = 0;
		rho = img_cur[read_index] - img_new[read_index_new];
		rho[3] = alpha;
	}
	Rho_[read_index] = rho;																			// save pixelwise photometric error map to buffer. NB Outside if(){}, to zero non-overlapping pixels.
	float4 rho_sq = {rho.x*rho.x,  rho.y*rho.y,  rho.z*rho.z, rho.w};
	local_sum_rho_sq[lid] = rho_sq;																	// Also compute global Rho^2.
	/*
	if (u<10 && v==10){
			printf("\nreduction=%u,  global_id_u=%u, u=%u, u_flt=%f, inv_depth=%f, uh2=%f, wh2=%f, u2_flt=%f, u2=%u,  rho=(%f,%f,%f,%f),  inv_depth=%f,  intersection=%f "\
					 ,reduction,     global_id_u,    u,    u_flt,    inv_depth,    uh2,    wh2,    u2_flt,    u2,     rho.x,rho.y,rho.z,rho.w,  inv_depth, intersection);
	}
	
	//if (intersection >0 ) {printf("\n  rho=(%f,%f,%f,%f)", rho.x,rho.y,rho.z,rho.w );}
	
	*/
	for (uint i=0; i<6; i++) {																		// for each SE3 DoF
		float8 SE3_grad_cur_px = SE3_grad_map_cur_frame[read_index     + i * mm_pixels ] ;
		float8 SE3_grad_new_px = SE3_grad_map_new_frame[read_index_new + i * mm_pixels ] ;
			
		float4 delta4;
		delta4.w=alpha;
		for (int j=0; j<3; j++) delta4[j] = rho[j] * (SE3_grad_cur_px[j] + SE3_grad_cur_px[j+4] + SE3_grad_new_px[j] + SE3_grad_new_px[j+4]);
			
		local_sum_grads[i*local_size + lid] = delta4;												// write grads to local mem for summing over the work group.
		SE3_incr_map_[read_index + i * mm_pixels ] = delta4;
	}
	////////////////////////////////////////////////////////////////////////////////////////		// Reduction
	int max_iter = 9;//ceil(log2((float)(group_size)));
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?  NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size){
			for (int i=0; i<num_DoFs; i++){
				local_sum_grads[i*local_size + lid] += local_sum_grads[i*local_size + lid + group_size];	// local_sum_grads  
			}
			local_sum_rho_sq[lid] += local_sum_rho_sq[lid + group_size];							// Also compute global Rho^2.
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid==0) {
		uint group_id 	= get_group_id(0);
		uint rho_global_sum_offset = read_offset_ / local_size ;									// Compute offset for this layer
		uint se3_global_sum_offset = rho_global_sum_offset *num_DoFs;								// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
		uint num_groups = get_num_groups(0);
		/*
		printf("\n__kernel se3_grad(..) reduction=%u,  se3_global_sum_offset=%u,  num_groups=%u,  group_id=%u, read_index=%u,  local_sum_rho_sq[lid]=(%f,%f,%f,%f)     local_sum_grads[lid]=", reduction, se3_global_sum_offset,  num_groups, group_id, read_index, local_sum_rho_sq[lid].x, local_sum_rho_sq[lid].y, local_sum_rho_sq[lid].z, local_sum_rho_sq[lid].w );
		for (int i=0; i<num_DoFs; i++){
			printf("\n  group_id=%u, i=%d, (%f, %f, %f, %f),  max_iter=%d",  group_id, i, local_sum_grads[i*local_size+lid].x, local_sum_grads[i*local_size+lid].y, local_sum_grads[i*local_size+lid].z, local_sum_grads[i*local_size+lid].w,  max_iter );
		}
		*/
		float4 layer_data = {num_groups, reduction, 0.0f, 0.0f };									// Write layer data to first entry
		if (global_id_u == 0) {
			global_sum_rho_sq[rho_global_sum_offset] 	= layer_data;
			global_sum_grads[se3_global_sum_offset] 	= layer_data; 
		}
		rho_global_sum_offset += 1 + group_id;
		se3_global_sum_offset += num_DoFs+ group_id*num_DoFs;
		
		if (local_sum_grads[0][3] >0){																// Using last channel local_sum_pix[0][7], to count valid pixels being summed.
			global_sum_rho_sq[rho_global_sum_offset] = local_sum_rho_sq[lid];
			for (int i=0; i<num_DoFs; i++){
				float4 temp_float4 = local_sum_grads[i*local_size + lid] / local_sum_grads[i*local_size + lid].w ;
				global_sum_grads[se3_global_sum_offset + i] = temp_float4 ;							// local_sum_grads  
			}																						// Save to global_sum_grads // Count hits, and divide group by num hits, without using atomics!
		}else {																						// If no matching pixels in this group, set values to zero.
			global_sum_rho_sq[rho_global_sum_offset] = 0;
			for (int i=0; i<num_DoFs; i++){
				global_sum_grads[rho_global_sum_offset] = 0;
			}
		}
		/*
		printf("\n global_sum_grads[%u] =  (%f,%f,%f,%f),  local_sum_rho_sq[%u] = (%f,%f,%f,%f)"\
		, rho_global_sum_offset, global_sum_grads[rho_global_sum_offset].x, global_sum_grads[rho_global_sum_offset].y, global_sum_grads[rho_global_sum_offset].z, global_sum_grads[rho_global_sum_offset].w \
		, lid,  local_sum_rho_sq[lid].x,  local_sum_rho_sq[lid].y,  local_sum_rho_sq[lid].z,  local_sum_rho_sq[lid].w  );
		*/
	}
}

__kernel void reduce (																				// TODO use this for the second stage image summation tasks.
	__constant 	uint*		mipmap_params,		//0		// kernel not currently in use, needs to integrate with mipmap.
	__constant 	uint*		uint_params,		//1
	__global	float8*		se3_sum,			//2
	__local		float8*		local_sum_grads,	//3
	__global 	float8*		se3_sum2			//4
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	if (global_id_u >= mipmap_params[MiM_PIXELS]) return;
	uint lid 			= get_local_id(0);
	uint local_size 	= get_local_size(0);
	uint group_size 	= local_size;
	uint read_offset_ 	= mipmap_params[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params[MiM_READ_COLS];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	
	uint global_sum_offset 	= read_offset_ / local_size ;											// Compute offset for this layer
	//
	local_sum_grads[lid] = se3_sum[global_sum_offset  + global_id_u];
	int max_iter = ilogb((float)(group_size));
	
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?  NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size)  local_sum_grads[lid] += local_sum_grads[lid+group_size];				// local_sum_grads  
	}
	
	if (lid==0) {
		uint group_id 	= get_group_id(0);
		uint global_sum_offset = read_offset_ / local_size ;										// Compute offset for this layer
		uint num_groups = get_num_groups(0);
		/*
		printf("\n\n reduction=%u,  global_sum_offset=%u,  num_groups=%u,  group_id=%u, \nlocal_sum_grads[lid]=( %f, %f, %f, %f,   %f, %f, %f, %f ),  \nse3_sum2[group_id]=( %f, %f, %f, %f,   %f, %f, %f, %f ) "\
		, reduction, global_sum_offset,  num_groups, group_id \
		, local_sum_grads[lid][0],local_sum_grads[lid][1],local_sum_grads[lid][2],local_sum_grads[lid][3], local_sum_grads[lid][4],local_sum_grads[lid][5],local_sum_grads[lid][6],local_sum_grads[lid][7] \
		, se3_sum2[group_id][0], se3_sum2[group_id][1], se3_sum2[group_id][2], se3_sum2[group_id][3],    se3_sum2[group_id][4], se3_sum2[group_id][5], se3_sum2[group_id][6], se3_sum2[group_id][7]\
		);
		*/
		float8 layer_data = {num_groups, reduction, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };			// Write layer data to first entry
		if (global_id_u == 0) {se3_sum2[global_sum_offset] = layer_data; }
		global_sum_offset += 1+ group_id;
		
		if (local_sum_grads[0][7] >0){
			se3_sum2[global_sum_offset] = local_sum_grads[0] / local_sum_grads[0][7];				// Save to se3_sum2 // Count hits, and divide group by num hits, without using atomics!
		}else se3_sum2[global_sum_offset] = 0;
	}
}

__kernel void convert_depth(
	__private	uint 	invert,					//0
	__private	float 	factor,					//1
	__constant 	uint*	mipmap_params,			//2		// NB uses ony mipmap_params[layer=0]
	__constant	uint*	uint_params,			//3
	__global	float* 	depth_mem,				//4
	__global	float* 	depth_mem_GT			//5
		)
{
	int global_id 		= (int)get_global_id(0);
	uint pixels 		= uint_params[PIXELS];
	
	uint read_offset_ 	= mipmap_params[MiM_READ_OFFSET];
	uint cols 			= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	
	uint base_row		= global_id/cols ;
	uint base_col		= global_id%cols ;
	uint img_row		= base_row + margin;
	uint img_col		= base_col + margin;
	
	uint read_index 	= read_offset_  +  base_row  * mm_cols  + base_col  ;
	uint global_id_u 	= get_global_id(0);
	
	if (global_id_u    >= mipmap_params[MiM_PIXELS]) return;
	float depth 		= depth_mem[global_id_u]/factor;
	
	//if (global_id_u == 0)printf("\n__kernel void convert_depth(..) invert=%u, factor=%f ", invert, factor );
	
	if (!(depth==0)){
		if ( invert==true ) depth_mem_GT[read_index] =  1/depth;
		else depth_mem_GT[read_index] = depth;
	}
}

__kernel void transform_depthmap(
	__private	uint	mipmap_layer,			//0
	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__global 	float16*k2k,					//3
	__global 	float4*	old_keyframe,			//4
	__global	float* 	depth_map_in,			//5
	__global	float* 	depth_map_out			//6
		)
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	
	uint local_size 	= get_local_size(0);
	uint group_size 	= local_size;
	uint work_dim 		= get_work_dim();
	uint global_size	= get_global_size(0);
	float16 k2k_pvt		= k2k[0];
	
	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels	= mipmap_params_[MiM_PIXELS];
	//if (global_id_u    >= layer_pixels) return;
	
	uint base_cols		= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint mm_pixels		= uint_params[MM_PIXELS];
	
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;													// read_row
	uint u 				= fmod(global_id_flt, read_cols_);											// read_column
	float u_flt			= u * reduction;															// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	float alpha			= old_keyframe[read_index].w;
	
	//uint depth_index	= v * reduction * base_cols + u * reduction;								// Sparse sampling of the depth map of the base image.
	float inv_depth 	= depth_map_in[read_index]; 	// depth_index								//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.//inv dept
	
	if (global_id_u ==0) printf("\n__kernel void transform_depthmap(): k2k_pvt= %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f, ",\
		k2k_pvt[0], k2k_pvt[1], k2k_pvt[2], k2k_pvt[3], k2k_pvt[4], k2k_pvt[5], k2k_pvt[6], k2k_pvt[7], k2k_pvt[8], k2k_pvt[9], k2k_pvt[10], k2k_pvt[11]
	);
	
	float uh2 = k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1 + k2k_pvt[3]*inv_depth;
	float vh2 = k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1 + k2k_pvt[7]*inv_depth;
	float wh2 = k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1+ k2k_pvt[11]*inv_depth;
	//float h/z  = k2k_pvt[12]*u_flt + k2k_pvt[13]*v + k2k_pvt[14]*1; // +k2k_pvt[15]/z
	
	float u2_flt		= uh2/wh2;
	float v2_flt		= vh2/wh2;
	int   u2			= floor((u2_flt/reduction)+0.5f) ;											// nearest neighbour interpolation
	int   v2			= floor((v2_flt/reduction)+0.5f) ;											// NB this corrects the sparse sampling to the redued scales.
	uint read_index_new = read_offset_ + v2 * mm_cols  + u2; // read_cols_
	if (global_id_u >= layer_pixels  || u2<0 || u2>read_cols_ || v2<0 || v2>read_rows_ ) return;
	////////////////////////////////////////////////////////
																									// TODO should I use more sophisticated interpolation ?
																									
	float newdepth = alpha * depth_map_in[read_index_new];	// alpha *								// old_keyframe.alpha indicates if this pixel of new depth map has valid source in the old depth map.
																									// could set a default value here.
	depth_map_out[read_index] = newdepth;// vh2/(read_rows_*256); 
	// uh2/(read_cols_*256); //v2_flt;// u2_flt;// vh2/read_cols_; // uh2/read_cols_; // ((float)read_index_new)/((float)mm_pixels); // newdepth; //depth_map_in[read_index]; //  
}

///////////////////// Photometrc cost ///////////////////

// Tau_HSV_grad(I) := distance in 8D space [ sin(Hue ), cos(Hue ), Saturation , (Saturation)/dx , (Saturation)/dy , Value , (Value)/dx , (Vallue)/dy ]
float Tau_HSV_grad (float8 B, float8 c){ // Poss also weight vector.
	float Tau_sq = 0;
	float8 Tau8 = B - c;
	for (int i = 0; i<8; i++){ Tau_sq += pown( Tau8[i], 2); }
	return sqrt(Tau_sq);
}

float8 Tau_HSV_grad_8chan (float8 B, float8 c){ // Poss also weight vector.
	float8 Tau8 = B - c;
	float8 absTau8;
	for (int i = 0; i<8; i++){ absTau8[i] = sqrt( pown( Tau8[i], 2) ); }  //      Tau_sq += pown( Tau8[i], 2);
	return Tau8;
}

///////////////////// Interpolation /////////////////////

float8 nearest_neigbour (float8* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction){
	float8 	c;
	int int_u2 = ceil(u_flt/reduction-0.5);									// nearest neighbour interpolation
	int int_v2 = ceil(v_flt/reduction-0.5);									// NB this corrects the sparse sampling to the redued scales.
	uint read_index_new = read_offset_ + int_v2 * cols  + int_u2;  			// uint read_index_new = (int_v2*cols + int_u2)*3;  // NB float4 c; float4* img now float8* for HSV_grad_mem
	c = img[read_index_new];
	return c;
}

float8 bilinear (__global float8* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction){  
	float8 	c, c_00, c_01, c_10, c_11;
	int coff_00, coff_01, coff_10, coff_11;
	int int_u2 = ceil(u_flt);
	int int_v2 = ceil(v_flt);
																			// compute adjacent pixel indices & sample adjacent pixels
	c_11 = img[ read_offset_ + int_v2     * cols +  int_u2     ];
	c_10 = img[ read_offset_ + (int_v2-1) * cols +  int_u2     ];
	c_01 = img[ read_offset_ + int_v2     * cols + (int_u2 -1) ];
	c_00 = img[ read_offset_ + (int_v2-1) * cols + (int_u2 -1) ];
																			// weighting for bi-linear interpolation
	float factor_x = fmod(u_flt,1);
	float factor_y = fmod(v_flt,1);
	c = factor_y * (c_11*factor_x  +  c_01*(1-factor_x))   +   (1-factor_y) * (c_10*factor_x  + c_00*(1-factor_x));
	return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////// Depth mapping //////////////////////////////////////////

__kernel void DepthCostVol(
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float16*k2k,				//4
	__global 	float8* base,				//5		keyframe_basemem  
	__global 	float8* img,				//6		HSV_grad_mem/*imgmem*/ now float8
	__global 	float*  cdata,				//7
	__global 	float*  hdata,				//8
	__global 	float*  lo,					//9
	__global 	float*  hi,					//10
	__global 	float*  a,					//11
	__global 	float*  d,					//12
	__global 	float*  img_sum,			//13
	__global 	float8* cdata_8chan			//14
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	
	uint8 mipmap_params_ = mipmap_params[mipmap_layer];									// choose correct layer of the mipmap
	
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_		= mipmap_params_[MiM_READ_ROWS];
	uint margin 		= uint_params[MARGIN];
	
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;										// read_row
	uint u 				= fmod(global_id_flt, read_cols_);								// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	/////////////////////////////////////////////////////////////
	/*
	if (global_id_u==0) printf("\n\n__kernel void DepthCostVol(): mipmap_layer=%i, mipmap_params_[MiM_PIXELS]=%i, mipmap_params_[MiM_READ_OFFSET]=%i, mipmap_params_[MiM_WRITE_OFFSET]=%i, mipmap_params_[MiM_READ_COLS]=%i, mipmap_params_[MiM_WRITE_COLS]=%i, mipmap_params_[MiM_GAUSSIAN_SIZE]=%i, mipmap_params_[MiM_READ_ROWS]=%i, mipmap_params_[MiM_WRITE_ROWS]=%i", mipmap_layer, mipmap_params_[MiM_PIXELS], mipmap_params_[MiM_READ_OFFSET], mipmap_params_[MiM_WRITE_OFFSET], mipmap_params_[MiM_READ_COLS], mipmap_params_[MiM_WRITE_COLS], mipmap_params_[MiM_GAUSSIAN_SIZE], mipmap_params_[MiM_READ_ROWS], mipmap_params_[MiM_WRITE_ROWS]
						);
	*/
	//int global_id = get_global_id(0);
	/*
	int pixels 			= floor(params[pixels_]);
	int rows 			= floor(params[rows_]);
	int cols 			= floor(params[cols_]);
	int layers			= floor(params[layers_]);
	float max_inv_depth = params[max_inv_depth_];  // not used
	float min_inv_depth = params[min_inv_depth_];
	float inv_d_step 	= params[inv_d_step_];
	float u 			= global_id % cols;														// keyframe pixel coords
	float v 			= (int)(global_id / cols);
	*/
	//int offset_3 		= global_id *3;													// Get keyframe pixel values
	float8 B = base[read_index];	//B.x = base[read_index].x;	B.y = base[read_index].y;	B.z = base[read_index].z;		// pixel from keyframe
	
	int costvol_layers	= uint_params[COSTVOL_LAYERS];
	//int pixels 			= uint_params[PIXELS];
	uint mm_pixels		= uint_params[MM_PIXELS];
	float inv_d_step 	= fp32_params[INV_DEPTH_STEP];
	float min_inv_depth = fp32_params[MIN_INV_DEPTH];
	
	float 	u2,	v2, rho,	inv_depth=0.0,	ns=0.0,	mini=0.0,	minv=3.0,	maxv=0.0;	// variables for the cost vol
	float8	rho_8chan;
	int 	int_u2, int_v2, coff_00, coff_01, coff_10, coff_11, cv_idx=read_index,	layer = 0;
	float8 	c, c_00, c_01, c_10, c_11;
	float 	c0 = cdata[cv_idx];															// cost for this elem of cost vol
	float 	w  = hdata[cv_idx];															// count of updates of this costvol element. w = 001 initially
	
	// layer zero, ////////////////////////////////////////////////////////////////////////////////////////
	// inf depth, rotation without paralax, i.e. reproj without translation.
	// Use depth=1 unit sphere, with rotational-preprojection matrix
	
	// precalculate depth-independent part of reprojection, h=homogeneous coords.
	float16 k2k_pvt		= k2k[0];
	float uh2 = k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1;   // +k2k[3]/z
	float vh2 = k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1;   // +k2k[7]/z
	float wh2 = k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1;  // +k2k[11]/z
	//float h/z  = k2k[12]*u_flt + k2k_pvt[13]*v_flt + k2k_pvt[14]*1; // +k2k[15]/z
	float uh3, vh3, wh3;
	/*
	if (global_id_u==0){ 
		printf("\n__kernel void DepthCostVol(): mipmap_layer=%i, k2k= ( %f,  %f,  %f,  %f, )( %f,  %f,  %f,  %f, )( %f,  %f,  %f,  %f, )( %f,  %f,  %f,  %f, )",\
		mipmap_layer, k2k_pvt[0], k2k_pvt[1], k2k_pvt[2], k2k_pvt[3], k2k_pvt[4], k2k_pvt[5], k2k_pvt[6], k2k_pvt[7], k2k_pvt[8], k2k_pvt[9], k2k_pvt[10], k2k_pvt[11], k2k_pvt[12], k2k_pvt[13], k2k_pvt[14], k2k_pvt[15] \
		);
		printf("\n__kernel void DepthCostVol(): uh2=%f, vh2=%f  , wh2=%f ", uh2, vh2, wh2 );
	}
	*/
	// cost volume loop  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	#define MAX_LAYERS 256 //64
	float cost[MAX_LAYERS];
	float8 cost_8chan[MAX_LAYERS];
	bool miss = false;
	bool in_image = false;
	if ( global_id_u  < mipmap_params_[MiM_PIXELS] ) in_image = true;
	
	for( layer=0;  layer<=costvol_layers; layer++ ){
		inv_depth = (layer * inv_d_step) + min_inv_depth;								// locate pixel to sample from  new image. Depth dependent part.
		uh3  = uh2 + k2k_pvt[3]*inv_depth;
		vh3  = vh2 + k2k_pvt[7]*inv_depth;
		wh3  = wh2 + k2k_pvt[11]*inv_depth;
		u2   = uh3/wh3;
		v2   = vh3/wh3;
		
		int_u2 = ceil(u2/reduction-0.5);												// nearest neighbour interpolation
		int_v2 = ceil(v2/reduction-0.5);												// NB this corrects the sparse sampling to the redued scales.
		/*
		//uint read_index_new = read_offset_ + int_v2 * mm_cols  + int_u2;  				// uint read_index_new = (int_v2*cols + int_u2)*3;  // NB float4 c; float4* img now float8* for HSV_grad_mem
		
		// read_index_new=%i,  read_index_new,
		if (global_id_u==0) printf("\n__kernel void DepthCostVol(): mipmap_layer=%i, read_offset_=%i, mm_pixels=%i,  cv_idx=%i  ###  depth layer=%i, inv_depth=%f, inv_d_step=%f,  min_inv_depth=%f,  uh3=%f,  vh3=%f,  wh3=%f,  u2=%f,  v2=%f,  int_u2=%i,  int_v2=%i  ", \
			mipmap_layer, read_offset_, mm_pixels,  cv_idx,    layer, inv_depth, inv_d_step, min_inv_depth, uh3, vh3, wh3, u2, v2, int_u2, int_v2 );
		*/
		if ( !((int_u2<0) || (int_u2>read_cols_ -1) || (int_v2<0) || (int_v2>read_rows_-1)) ) {  	// if (not within new frame) skip     || (in_image == false)
			cv_idx = read_index + layer*mm_pixels;											// Step through costvol layers
			cost[layer] = cdata[cv_idx];													// cost for this elem of cost vol
			w  = hdata[cv_idx];																// count of updates of this costvol element. w = 001 initially
			
			// c = img[read_index_new];																// nearest neighbour
			c = bilinear(img, u2/reduction, v2/reduction, mm_cols, read_offset_, reduction); 		// bilinear(float8* img, float u_flt, float v_flt, int cols)
			/*
			float rx		= (c.x-B.x);   float ry= (c.y-B.y);   float rz= (c.z-B.z);	// Compute photometric cost // L2 norm between keyframe & new frame pixels.
			rho 			= sqrt( rx*rx + ry*ry + rz*rz )*50;							//TODO make *50 an auto-adjusted parameter wrt contrast in area of interest.
			*/
			/*
			for (int r_n=0; r_n<4; r_n++) {rho += pown(r[r_n],2);}
			//rho += pown(r[3],2);
			rho = sqrt(rho);
			...
			float8 r 		= c-B;
			*/
			rho						= Tau_HSV_grad(B, c);								// Compute rho photometic cost
			rho_8chan				= Tau_HSV_grad_8chan(B, c);
			cost[layer] 			= (cost[layer]*w + rho) / (w + 1);	 				// Compute update of cost vol element, taking account of 'w  = hdata[cv_idx];' number of hits to this element.
			cost_8chan[layer] 		= (cost_8chan[layer]*w + rho_8chan) / (w + 1);	
		
			cdata[cv_idx] 			= cost[layer];  									// CostVol set here ###########
			cdata_8chan[cv_idx] 	= cost_8chan[layer];
			hdata[cv_idx] 	= w + 1;													// Weightdata, counts updates of this costvol element.
			img_sum[cv_idx] += (c.x + c.y + c.z)/3;
		} else { miss = true; }
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if ( (miss == false) && (in_image == true) ) { 
		for( layer=0;  layer<costvol_layers; layer++ ){
			if (cost[layer] < minv) { 														// Find idx & value of min cost in this ray of cost vol, given this update.
				minv = cost[layer];															// NB Use array private to this thread.
				mini = layer;
			}
			maxv = fmax(cost[layer], maxv);
		}
		lo[read_index] 	= minv; 															// min photometric cost  // rho;//
		a[read_index] 	= mini*inv_d_step + min_inv_depth; //c.x; //uh2; //c.x; // mini*inv_d_step + min_inv_depth;	// inverse distance
		d[read_index] 	= mini*inv_d_step + min_inv_depth; //B.x; //mini*inv_d_step + min_inv_depth; //uh3; //c.y; // mini*inv_d_step + min_inv_depth; 
		hi[read_index] 	= maxv; 															// max photometric cost
	}
}

__kernel void UpdateQD(
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float4* g1pt,				//4
	__global 	float* 	qpt,				//5		// qmem,						//	2 * mm_size_bytes_C1
	__global 	float*  apt,				//6		// amem,     auxilliary A
	__global 	float*  dpt					//7		// dmem,     depth D
	//__global 	float* 	qpt2				//8		// qmem,						//	2 * mm_size_bytes_C1
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	
	uint8 mipmap_params_ = mipmap_params[mipmap_layer];					// choose correct layer of the mipmap
	uint mim_pixels		= mipmap_params_[MiM_PIXELS];					// cannot return before the last memory barrier !
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_		= mipmap_params_[MiM_READ_ROWS];
	uint margin 		= uint_params[MARGIN];
	
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;						// read_row
	uint u 				= fmod(global_id_flt, read_cols_);				// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	
	//uint mm_pixels		= uint_params[MM_PIXELS];
	///////////////////////////////////
	
	//int g_id 			= get_global_id(0);
	//int rows 			= floor(params[rows_]);
	//int cols 			= floor(params[cols_]);
	int costvol_layers	= uint_params[COSTVOL_LAYERS];
	float epsilon 		= fp32_params[EPSILON];
	float sigma_q 		= fp32_params[SIGMA_Q];
	float sigma_d 		= fp32_params[SIGMA_D];
	float theta 		= fp32_params[THETA];
	
	int y = global_id_u / read_cols_;
	int x = global_id_u % read_cols_;
	unsigned int pt = read_index ;										//x + y * mm_cols;					// index of this pixel
	
	const int wh = uint_params[MM_PIXELS]; 								//(mm_pixels + read_offset_);		//  / *mm_cols*read_rows_* /
	
/*	
 * //	barrier(CLK_GLOBAL_MEM_FENCE);
	if (global_id_u==0){
		printf("\n Kernel UpdateQD  mipmap_params_[MiM_PIXELS]=%i, reduction=%i, pt=%i, wh=%i,  pt+wh=%i,  x=%i,  y=%i, read_offset_=%i, (mm_cols*read_rows_ + read_offset)=%i,     v=%i,   mm_cols=%i,   u=%i,   read_cols_=%i    ",\
													mipmap_params_[MiM_PIXELS], reduction,	 pt, 	wh,		 pt+wh, 	x, 		y, 	read_offset_, 	(mm_cols*read_rows_ + read_offset_),        v,      mm_cols,      u,      read_cols_       );  
		//printf("\n sigma_q*epsilon = %f,   sigma_q=%f,  epsilon=%f", sigma_q*epsilon, sigma_q, epsilon);
	}	
	// if (pt >= (mm_cols*read_rows_ + read_offset_)) printf("\n Kenel UpdateQD   pt=%i, wh=%i,  pt+wh=%i,  x=%i,  y=%i",pt, wh, pt+wh, x, y ); return;
	
	//if (hdata[pt+ (costvol_layers-1)*rows*cols] <=0.0) return;		// if no input image overlaps, on layer 0 of hitbuffer, skip this pixel. // makes no difference
	
	//if (pt == (mm_cols*read_rows_ + read_offset_) -1 ) printf("\n Kenel UpdateQD   pt=%i, wh=%i,  pt+wh=%i,  x=%i,  y=%i",pt, wh, pt+wh, x, y );
*/
	float4 g1_4;
	float g1, qx, qy, d, a;
	float dd_x, dd_y, maxq;
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if (global_id_u < mim_pixels) {
		g1_4 = g1pt[pt];
		g1 =  g1_4.x * g1_4.y * g1_4.z ;									// reduce channel count of g1. Here Manhatan norm. bad choice. Hue is not good. 
		qx = qpt[pt];														// TODO Later try   g1 = 1-(1-g_saturation)*(1-g_value) , i.e. where sat and val agree: less fooled by shadows.
		qy = qpt[pt+wh];
		d  = dpt[pt];
		a  = apt[pt];
		
		dd_x = (x==read_cols_-1)? 0.0f : dpt[pt+1]       - d;				// Sample depth gradient in x&y
		dd_y = (y==read_rows_-1)? 0.0f : dpt[pt+mm_cols] - d;				// dd_x, dd_y = 0, if at edge of image, otherwise depth_grad in x,y. 
		
		// qpt2[pt] 		= dd_x;		//dpt[pt+1]			- d;  //x;	//dd_x;		//global_id_u;  //
		// qpt2[pt + wh] 	= dd_y;		//dpt[pt+mm_cols]	- d;		//dd_y;		//pt;			//
		
		qx = (qx + sigma_q*g1*dd_x) / (1.0f + sigma_q*epsilon);				// DTAM paper, primal-dual update step
		qy = (qy + sigma_q*g1*dd_y) / (1.0f + sigma_q*epsilon);				// sigma_q=0.0559017,  epsilon=0.1,  g1=0.999.. if white, less if visible edge.
		maxq = fmax(1.0f, sqrt(qx*qx + qy*qy));
		
		if (x==100 && y==100) printf("\nKernel UpdateQD_1 mipmap_layer=%u, mim_pixels=%u, mm_cols=%u, wh=%u, pt=%u, d=%f, sigma_q=%f, epsilon=%f, g1=%f, , a=%f, theta=%f, sigma_d=%f, qx=%f, qy=%f, maxq=%f, dd_x=%f, dd_y=%f m x=%i, y=%i", \
			mipmap_layer, mim_pixels, mm_cols, wh, pt, d, sigma_q, epsilon, g1,  a, theta, sigma_d, qx, qy, maxq, dd_x, dd_y, x, y );
		
		qx 			= qx/maxq;
		qy 			= qy/maxq;
		
		qpt[pt]		= qx;  													//dd_x;//pt2;//wh;//pt;//dd_x;//qx / maxq;
		qpt[pt+wh]	= qy;  													//dd_y;//pt;//;//y;//dd_y;//dpt[pt+1] - d; //dd_y;//qy / maxq;
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);											// needs to be after all Q updates.
	if (global_id_u < mim_pixels){
		float dqx_x;														// = div_q_x(q, w, x, i);				// div_q_x(..)
		if (x == 0) dqx_x = qx;
		else if (x == read_cols_-1) dqx_x =  -qpt[pt-1];
		else dqx_x =  qx- qpt[pt-1];
		
		float dqy_y;														// = div_q_y(q, w, h, wh, y, i);		// div_q_y(..)
		if (y == 0) dqy_y =  qy;											// return q[i];
		else if (y == read_rows_-1) dqy_y = -qpt[pt+wh-mm_cols];			// return -q[i-1];
		else dqy_y =  qy - qpt[pt+wh-mm_cols/*read_cols_*/];							// return q[i]- q[i-w];
		
		const float div_q = dqx_x + dqy_y;
		
		dpt[pt] = (d + sigma_d * (g1*div_q + a/theta)) / (1.0f + sigma_d/theta);
		
		if (x==100 && y==100) printf("\nKernel UpdateQD_2 mipmap_layer=%u, mm_cols=%u, wh=%u, dpt[pt]=%f, d=%f, sigma_q=%f, epsilon=%f, g1=%f, div_q=%f, a=%f, theta=%f, sigma_d=%f, qx=%f, qy=%f, maxq=%f, dd_x=%f, dd_y=%f ", \
			mipmap_layer, mm_cols, wh, dpt[pt], d, sigma_q, epsilon, g1, div_q , a, theta, sigma_d, qx, qy, maxq, dd_x, dd_y );
	}
}

__kernel void  UpdateG(
	__private	uint		mipmap_layer,	//0
	__constant	uint8*		mipmap_params,	//1
	__constant 	uint*		uint_params,	//2
	__constant 	float*		fp32_params,	//3
	__global 	float4*		img,			//4 
	__global 	float4*		g1p				//5
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	
	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint mm_cols		= uint_params[MM_COLS];
	uint read_row    	= global_id_u / read_cols_;
	uint read_column 	= fmod(global_id_flt, read_cols_);
	
	int upoff			= -(read_row  >1 )*mm_cols;													//-(read_row  != 0)*mm_cols;				// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;
    int lfoff			= -(read_column >1);														//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);											// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;

	float alphaG		= fp32_params[ALPHA_G];
	float betaG 		= fp32_params[BETA_G];
	
	float4 pr 	= img[offset + rtoff];
	float4 pl 	= img[offset + lfoff];
	float4 pu 	= img[offset + upoff];
	float4 pd 	= img[offset + dnoff];

	float4 gx	= { (pr.x - pl.x), (pr.y - pl.y), (pr.z - pl.z), 1.0f };							// Signed img gradient in hsv
	float4 gy	= { (pd.x - pu.x), (pd.y - pu.y), (pd.z - pu.z), 1.0f };
	 
	float4 g1	= { \
		 exp(-alphaG * pow(sqrt(gx.x*gx.x + gy.x*gy.x), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.y*gx.y + gy.y*gy.y), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.z*gx.z + gy.z*gy.z), betaG) ), \
		 1.0f };
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;	 
	g1p[offset]= g1;
}

int set_start_layer(float di, float r, float far, float depthStep, int layers, int x, int y){ //( inverse_depth, r , min_inv_depth, inv_depth_step, num_layers )
    const float d_start = di - r;
    const int start_layer =  floor( (d_start - far)/depthStep );
    return (start_layer<0)? 0 : start_layer;
}

int set_end_layer(float di, float r, float far, float depthStep, int layers, int x, int y){
    const float d_end = di + r;
    const int end_layer = ceil((d_end - far)/depthStep) + 1;
    return  (end_layer>(layers-1))? (layers-1) : end_layer;
}

float get_Eaux(float theta, float di, float aIdx, float far, float depthStep, float lambda, float scale_Eaux, float costval)
{
	const float ai = far + aIdx*depthStep;
	return scale_Eaux*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	/*
	// return (0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 100*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 10000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 1000000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 10000000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	*/
}

__kernel void UpdateA(						// pointwise exhaustive search
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float*  cdata,				//4		// cdatabuf, cost volume
	__global 	float*  lo,					//5
	__global 	float*  hi,					//6
	__global 	float*  apt,				//7		// amem,     auxilliary A
	__global 	float*  dpt					//8		// dmem,     depth D
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	
	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	//if (global_id_u    >= mipmap_params_[MiM_PIXELS]) return;
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;					// read_row
	uint u 				= fmod(global_id_flt, read_cols_);			// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	//////////////////////////////////////
	
	//int x 				= get_global_id(0);						// int costvol_layers	= uint_params[COSTVOL_LAYERS];
	//int rows 				= floor(params[rows_]);
	//int cols 				= floor(params[cols_]);
	int costvol_layers		= uint_params[COSTVOL_LAYERS];			//floor(params[layers_]);
	unsigned int layer_step = uint_params[MM_PIXELS];				//floor(params[pixels_]);
	float lambda			= fp32_params[LAMBDA];					//params[lambda_];
	float theta				= fp32_params[THETA];					//params[theta_];
	float max_d				= fp32_params[MAX_INV_DEPTH];			//params[max_inv_depth_]; //near
	float min_d				= fp32_params[MIN_INV_DEPTH];			//params[min_inv_depth_]; //far
	float scale_Eaux		= fp32_params[SCALE_EAUX];				//params[scale_Eaux_];
	/*
	int y = global_id_u / read_cols_;
	x = x % read_cols_;
	unsigned int pt  = x + y * mm_cols * margin;					// index of this pixel
	if (pt >= (cols*rows))return;
	//if (hdata[pt+ (costvol_layers-1)*rows*cols] <=0.0) return;	// if no input image overlaps, on layer 0 of hitbuffer, skip this pixel.// makes no difference
	*/
	unsigned int cpt = read_index;
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (global_id_u    >= mipmap_params_[MiM_PIXELS]) return;
	
	float d  		= dpt[read_index];
	float E_a  		= FLT_MAX;
	float min_val	= FLT_MAX;
	int   min_layer	= 0;
	
	const float depthStep 	= fp32_params[INV_DEPTH_STEP];			//(min_d - max_d) / (costvol_layers - 1);
	//const int   layerStep 	= rows*cols;
	const float r 			= sqrt( 2*theta*lambda*(hi[read_index] - lo[read_index]) );
	const int 	start_layer = set_start_layer(d, r, max_d, depthStep, costvol_layers, u, v);  // 0;//
	const int 	end_layer   = set_end_layer  (d, r, max_d, depthStep, costvol_layers, u, v);  // costvol_layers-1; //
	int 		minl 		= 0;
	float 		Eaux_min 	= 1e+30; 								// set high initial value
	
	for(int l = start_layer; l <= end_layer; l++) {
		const float cost_total = get_Eaux(theta, d, (float)l, min_d, depthStep, lambda, scale_Eaux, cdata[read_index+l*layer_step]);
		// apt[read_index+l*layerStep] = cost_total;  				// DTAM_Mapping collects an Eaux volume, for debugging.
		if(cost_total < Eaux_min) {
			Eaux_min = cost_total;
			minl = l;
		}
	 }
	float a = min_d + minl*depthStep;  								// NB implicit conversion: int minl -> float.

	//refinement step
	if(minl > start_layer && minl < end_layer){ 					//return;// if(minl == 0 || minl == costvol_layers-1) // first or last was best
																	// sublayer sampling as the minimum of the parabola with the 2 points around (minl, Eaux_min)
		const float A = get_Eaux(theta, d, minl-1, max_d, depthStep, lambda, scale_Eaux, cdata[read_index+(minl-1)*layer_step]);
		const float B = Eaux_min;
		const float C = get_Eaux(theta, d, minl+1, max_d, depthStep, lambda, scale_Eaux, cdata[read_index+(minl+1)*layer_step]);
		// float delta = ((A+C)==2*B)? 0.0f : ((A-C)*depthStep)/(2*(A-2*B+C));
		float delta = ((A+C)==2*B)? 0.0f : ((C-A)*depthStep)/(2*(A-2*B+C));
		delta = (fabs(delta) > depthStep)? 0.0f : delta;
		// a[i] += delta;
		a -= delta;
	}
	apt[read_index] = a;

	//if (x==200 && y==200) printf("\n\nUpdateA: theta=%f, lambda=%f, hi=%f, lo=%f, r=%f, d=%f, min_d=%f, max_d=%f, minl=%f, depthStep=%f, costvol_layers=%i, start_layer=%i, end_layer=%i", \
	//	theta, lambda, hi[read_index], lo[read_index], r, d, min_d, max_d, minl, depthStep, costvol_layers, start_layer, end_layer );
}

////////// Non-Primal-Dual kernels for cost functions //////////////////////////////////////////////////////////////////////////
/// Priors on Shape ////
__kernel void inv_depth_grad (
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float*  inv_depth,			//4		// dmem,     depth 
	__global 	float2*  grad_inv_depth		//5
){
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	
	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint /*v*/  read_row  	= global_id_u / read_cols_;					// read_row
	uint /*u*/  read_column	= fmod(global_id_flt, read_cols_);			// read_column
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column ;
	////
	// From __kernel void  img_grad(..)
	int upoff			= -(read_row  >1 )*mm_cols;													//-(read_row  != 0)*mm_cols;				// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;
    int lfoff			= -(read_column >1);														//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);											// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;
	
	float pr =  inv_depth[offset + rtoff];
	float pl =  inv_depth[offset + lfoff];
	float pu =  inv_depth[offset + upoff];
	float pd =  inv_depth[offset + dnoff];
	
	float2 grad;
	grad.x = pr - pl;
	grad.y = pd - pu;
	
	grad_inv_depth[read_index]	= grad;
}


__kernel void div_inv_depth_grad (	// Divergence of gradient of inverse depth i.e. smoothness of gradient of depth
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float2* grad_inv_depth,		//5
	__global 	float*  div_inv_depth_grad,	//6
	__global 	float2* grad2_inv_depth		//7
)
{
	// from __kernel void UpdateA(..)
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	
	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint /*v*/  read_row   	= global_id_u / read_cols_;					// read_row
	uint /*u*/  read_column = fmod(global_id_flt, read_cols_);			// read_column
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column ;
	///////////////////////////
	// Definition of Divergence of a vector field v:  Del dot v(x,y) = gradient_v(x,z) dot v(x,y)  = delta(V_x) / delta(x)  +  delta(V_y) / delta(y)
	
	// From __kernel void  img_grad(..)
	int upoff			= -(read_row  >1 )*mm_cols;													//-(read_row  != 0)*mm_cols;				// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;
    int lfoff			= -(read_column >1);														//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);											// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;
	
	float pr =  grad_inv_depth[offset + rtoff].x;
	float pl =  grad_inv_depth[offset + lfoff].x;
	float pu =  grad_inv_depth[offset + upoff].y;
	float pd =  grad_inv_depth[offset + dnoff].y;
	
	float2 grad;
	grad.x = pr - pl;
	grad.y = pd - pu;
	
	grad2_inv_depth[read_index]		= grad;
	div_inv_depth_grad[read_index]	= (grad_inv_depth[read_index].x  *  grad.x ) +  (grad_inv_depth[read_index].y   *  grad.y ) ;
}


__kernel void project_point_cloud (				// Possibly useful output, saves CPU work.
	__private	uint		mipmap_layer,		//0
	__constant 	uint8*		mipmap_params,		//1
	__constant 	uint*		uint_params,		//2
	__global 	float16*	inv_k,				//3		// inverse projection matrix
	__global 	float*  	fp32_params,		//4
	__global 	float*  	inv_depth,			//5
	__global 	float3* 	point_cloud			//6
)
{
	// from __kernel void DepthCostVol(..)
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint8 mipmap_params_= mipmap_params[mipmap_layer];									// choose correct layer of the mipmap
	
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint margin 		= uint_params[MARGIN];
	
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;										// read_row
	uint u 				= fmod(global_id_flt, read_cols_);								// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	//////////////
	
	uint mm_pixels		= uint_params[MM_PIXELS];
	if (global_id_u > mm_pixels) return;
	
	float inv_d_step 	= fp32_params[INV_DEPTH_STEP];
	float min_inv_depth = fp32_params[MIN_INV_DEPTH];
	
	float16 inv_k_pvt		= inv_k[0];
	float 	inv_depth_pvt 	= inv_depth[read_index];
	float3 	point;
	
	point.x = inv_k_pvt[0]*u_flt + inv_k_pvt[1]*v_flt + inv_k_pvt[ 2]*1 + inv_k_pvt[ 3]*inv_depth_pvt;  // TODO check correct matrix computaion for inverse projection. esp 3rd & 4th cols of this equation.
	point.y = inv_k_pvt[4]*u_flt + inv_k_pvt[5]*v_flt + inv_k_pvt[ 6]*1 + inv_k_pvt[ 7]*inv_depth_pvt;
	point.z = inv_k_pvt[8]*u_flt + inv_k_pvt[9]*v_flt + inv_k_pvt[10]*1 + inv_k_pvt[11]*inv_depth_pvt;
	
	point_cloud[read_index] = point;
}


__kernel void compute_curvature (
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float*  inv_depth,			//4	
	__global 	float*  mean_curvature,		//5
	__global	float8* loss_params			//6
)
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint8 mipmap_params_= mipmap_params[mipmap_layer];									// choose correct layer of the mipmap
	
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_pixels		= uint_params[MM_PIXELS];
	if (global_id_u > mm_pixels) return;
	
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint /*v*/  read_row	= global_id_u / read_cols_;									// read_row
	uint /*u*/  read_column	= fmod(global_id_flt, read_cols_);							// read_column
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column ;
	
	// From SIRFS : Section 5.1 ... Mean curvature is defined as the average of principal curvatures.
	// H = 1/2(k_1 + k_2) , can be approxiated on a surface using filter convolutions that prroximate 1st & 2nd partial derivatives.
	// Eq 15.
	// H(Z) = ((1+Z²_x) z_yy - 2Z_x*Z_y*Z_xy +(1+Z²_y)Z_xx)  / (2(1+Z²_x + Z2_y)⁽3/2) )
	
	// From SIRFS supplementary data Section 3
	// Mean curvature H(Z) of depthmap Z.
	
	int upoff			= -(read_row  >1 )*mm_cols;										//-(read_row  != 0)*mm_cols;				// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;						// (read_row  < read_rows_-1) * mm_cols;
	int lfoff			= -(read_column >1);											//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);								// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;
	
	float a = inv_depth[offset+lfoff+upoff];
	float b = inv_depth[offset		+upoff];
	float c = inv_depth[offset+rtoff+upoff];
	float d = inv_depth[offset+lfoff];
	float e = inv_depth[offset];
	float f = inv_depth[offset+rtoff];
	float g = inv_depth[offset+lfoff+dnoff];
	float h = inv_depth[offset		+dnoff];
	float i = inv_depth[offset+rtoff+dnoff];
	 
	float Z_x  = ( 2*(d - f) + (a - c) + (g - i) )/8;
	float Z_y  = ( 2*(c - h) + (a - g) + (c - i) )/8;
	float Z_xx = ( a -2*b + c + 2*d -4*e + 2*f + g - 2*h + i )/4;
	float Z_yy = ( a +2*b + c - 2*d -4*e - 2*f + g + 2*h + i )/4;
	float Z_xy = ( a - c + g - i )/4;
	
	float Z_x_sq = pown(Z_x,2);
	float Z_y_sq = pown(Z_y,2);
	
	float M = sqrt(1 + Z_x_sq + Z_y_sq );
	float N = ((1 + Z_x_sq) * Z_yy) - (2*Z_x*Z_y*Z_xy) + ((1 + Z_y_sq) * Z_xx);
	float D = 2 * pown(M,3);
	
	mean_curvature[offset] = N/D;
	
	float M_sq = pown(M,2);
	float F_x  = 2*(Z_x*Z_yy - Z_xy*Z_y) - (3*Z_x*N)/M_sq;
	float F_y  = 2*(Z_xx*Z_y - Z_x*Z_xy) - (3*Z_y*N)/M_sq;
	float F_xx = 1 + Z_y_sq;
	float F_yy = 1 + Z_x_sq;
	float F_xy = -2 *Z_x*Z_y;
	
	float8 loss_params_pvt = {M, N, D, F_x, F_y, F_xx, F_yy, F_xy};						// M & N included for debugging.
	
	loss_params[offset] = loss_params_pvt;
}

// NB SIRFS section 5.1 "Our smoothness prior for shapes is a Gaussian scale mixture on the localvariaion of the mean curvature of Z (the depth map).
// Eq 16.  f_k(Z) = SUM_i{ SUM_(j in N(i)){ c(H(Z)_i - H(Z)_j  , alpha_k, sigma_k) } }
// NB this produces a distribution of cost for +ve and -ve curvatures, with minimum cost for a plane surface.
// The learned GSM is very heavy tailed, fig 11, encourages mostly smooth, with occassionally very non-smooth, i.e. bend rarely.

__kernel void curvature_gradient(
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float*  mean_curvature,		//4
	__global 	float*  grad_curvature		//5
			  )
{
	
	// TODO ? should we rather have geeric kernel code for grad & div, then in host "createKernel(...)" different instances, with different Arguments set, for inv_depth, curvature, and other maps. // 
	
	
}

__kernel void curvature_smoothness (
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float*  mean_curvature,		//4
	__global 	float*  div_curvature		//5
)
{
	
	
	
	
}
													// Parsimony costs : Need to sort pixels by bin-sort, then attraction between neighbours #################################################
__kernel void plane_parsimony ( 					// 3D orientation of plane + orthogonal distance from camera
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{
	
}

__kernel void curvature_parsimony ( 				// 2D major,minor curvature
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{
	
}

/// Priors on Reflectance ///////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void reflectance_smoothness ( 				// Modfied Cook-Torrance,  HSV_lambertian, HSV_specular, roughness, metal-glass  (8D reflectance space) NB Plenoxels transmitance....
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{
	
}

__kernel void reflectance_parsimony ( 
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{
	
}

__kernel void reflectance_absolute_value ( 
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{
	
}

__kernel void illumination_model ( 
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{
	
}

//// relative velocity map /////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void smoothness_relative_velocity ( 
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{
	
}

//// rendering model //////////// 					// called (i) to construct cost vol (ii) predict next frame.  NB need modfied Cook-Torrance with glassy & metalic specular components.

__kernel void render_image ( 
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{
	
}







////////////////////////////////////////////////////////////////////////////////////

__kernel void MeasureDepthFit(						// measure the fit of the depthmap against the groud truth.
	__private	uint	mipmap_layer,				//0
	__constant 	uint8*	mipmap_params,				//1
	__constant 	uint*	uint_params,				//2
	__global 	float*  fp32_params,				//3
	__global 	float*  dpt,						//4		// dmem,     depth D
	__global 	float*  dpt_GT,						//5	
	__global 	float4* dpt_disparity,				//6	
	__local		float4*	local_sum_dpt_disparity,	//7
	__global	float4*	global_sum_dpt_disparity	//8
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	
	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;					// read_row
	uint u 				= fmod(global_id_flt, read_cols_);			// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	
	float depth_disparity = dpt[read_index] - dpt_GT[read_index];
	
	float sq_depth_disparity = depth_disparity * depth_disparity * 64 * 64;
	
	float proportional_sdd = sq_depth_disparity / (dpt_GT[read_index] * dpt_GT[read_index] * 64 * 8);
	
	float4 disparity 	= { depth_disparity , -depth_disparity, proportional_sdd, 1.0f};
							
						//= {dpt_GT[read_index] * 64, sq_depth_disparity, proportional_sdd , 1.0f };
																							// B = true inverse depth
																							// G = sq_depth_disparity
																							// R = proportional_sdd
						//= {dpt_GT[read_index], depth_disparity , -depth_disparity, 1.0f }; 
																							// {B, G, R, A} Need x64 to spread in visible range of .tiff .
																							// B = true inverse depth
																							// G = est_inv_depth > true inv_depth, i.e. est too close
																							// R = est_inv_depth < true inv_depth, i.e. est too far
	if (global_id_u    < mipmap_params_[MiM_PIXELS]){
		dpt_disparity[read_index]		= disparity;
		local_sum_dpt_disparity[lid]	= disparity;
	}else{
		local_sum_dpt_disparity[lid]	= 0;
	}

	int max_iter = ilogb((float)(group_size));
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)					// problem : how to produce one result for each mipmap layer ?  
																										// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																	// No 'if->return' before fence between write & read local mem
		
		if (lid<group_size)  local_sum_dpt_disparity[lid] += local_sum_dpt_disparity[lid+group_size];	// local_sum_pix  
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);  // TODO get summation of depth error working 
	if (lid==0) {
		uint group_id 			= get_group_id(0);
		uint global_sum_offset 	= 0; 																	//read_offset_ / local_size ;		// only the base layer		// Compute offset for this layer
		uint num_groups 		= get_num_groups(0);

		float4 layer_data 		= {num_groups, reduction, 0.0f, 0.0f };									// Write layer data to first entry  ## problem float4 into float ##
		if (global_id_u == 0) 	{
			global_sum_dpt_disparity[global_sum_offset] 	= num_groups; 
			global_sum_dpt_disparity[global_sum_offset+1] 	= reduction;
			global_sum_dpt_disparity[global_sum_offset+2] 	= 0.0f;
			global_sum_dpt_disparity[global_sum_offset+3] 	= 0.0f;
		}
		global_sum_offset += 4 + group_id;
		
		if (local_sum_dpt_disparity[0][3] >0){																	// Using alpha channel local_sum_pix[0][3], to count valid pixels being summed.
			global_sum_dpt_disparity[global_sum_offset] 	= local_sum_dpt_disparity[0];				// Save to global_sum_pix // Count hits, and divide group by num hits, without using atomics!
		}else global_sum_dpt_disparity[global_sum_offset] 	= 0;
	}
}


/*
//uint margin 		= uint_params[MARGIN];
	//uint v    			= global_id_u / read_cols_;					// read_row
	//uint u 				= fmod(global_id_flt, read_cols_);			// read_column
	//float u_flt			= u * reduction;
	//float v_flt			= v * reduction;
	//uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	
	/////
	//int max_iter = ilogb((float)(group_size));
	//for (uint iter=0; iter<max_iter ; iter++) {	// for log2(local work group size)		// problem : how to produce one result for each mipmap layer ?  NB kernels launched separately for each layer, but workgroup size varies between GPUs.
	//	group_size   /= 2;
	//	if (lid>group_size)  return;
	//	local_sum_grads[lid] += local_sum_grads[lid+group_size];						// local_sum_grads  
	//}
	
	//uint group_id 	= get_group_id(0);
	//se3_sum[group_id] = local_sum_grads[0];										// save to global_sum_grads  //  TODO ##### this will clash if work groups do not finish in order !!!!!!!!!!!!!
	//printf("\nreduction=%u,  global_id_u=%u,   group_id=%u,  lid=%u,  local_sum_grads[0]=(%f,%f,%f,%f,  %f,%f,%f,%f,  %f,%f,%f,%f,  %f,%f,%f,%f) ", reduction, global_id_u, group_id, lid  \
	//	, local_sum_grads[0][0], local_sum_grads[0][1], local_sum_grads[0][2], local_sum_grads[0][3],     local_sum_grads[0][4], local_sum_grads[0][5], local_sum_grads[0][6], local_sum_grads[0][7] \
	//	, local_sum_grads[0][8], local_sum_grads[0][9], local_sum_grads[0][10], local_sum_grads[0][11],     local_sum_grads[0][12], local_sum_grads[0][13], local_sum_grads[0][14], local_sum_grads[0][15] \
	//);
*/


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
	__constant 	uint*	mipmap_params,	//0
	__constant 	uint*	uint_params,	//1
	
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	if (global_id_u >= mipmap_params[MiM_PIXELS]) return;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	uint read_offset_ 	= mipmap_params[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params[MiM_READ_COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;					// read_row
	uint u 				= fmod(global_id_flt, read_cols_);			// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	///
	
	
	
	
}
__kernel void  (
	__constant 	uint*	mipmap_params,	//0
	
		 )
{
	
	
}
__kernel void  (
	__constant 	uint*	mipmap_params,	//0
	
		 )
{
	
	
}

*/

