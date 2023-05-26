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
#define SE3_LM_A			11	// LM damped least squares parameters for SE3 tracking
#define SE3_LM_B			12

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
#define MiM_READ_ROWS		6	// rows without margins
#define MiM_WRITE_ROWS		7

__kernel void cvt_color_space_linear(							// Writes the first entry in a linear mipmap
	__global	uchar*	base,			//0
	__global	float4*	img,			//1
	__constant	uint*	uint_params		//2
		 )
{																// NB need 32-bit uint (2**32=4,294,967,296) for index, not 16bit (2**16=65,536).
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
	
	 if (!(H<=1.0f && H>=0.0f) || !(S<=1.0f && S>=0.0f) || !(V<=1.0f && V>=0.0f) ) {H=S=V=0.0f;}	// to replace any NaNs
	 
	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	uint img_row	= base_row + margin;
	uint img_col	= base_col + margin;
	uint img_index	= img_row*(cols + 2*margin) + img_col;   										// NB here use cols not mm_cols
	
	float4 temp_float4  = {H,S,V,1.0f};																// Note how to load a float4 vector.
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

__kernel void mipmap_linear_flt(																	// Nvidia Geforce GPUs cannot use "half"
	__constant 	uint*	mipmap_params,	//0
	__constant 	float* 	gaussian,		//1
	__constant 	uint*	uint_params,	//2
	__global 	float4*	img,			//3
	__local	 	float4*	local_img_patch //4
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	uint patch_length	= group_size+2;
	
	uint read_offset_ 	= 1*mipmap_params[MiM_READ_OFFSET];
	uint write_offset_ 	= 1*mipmap_params[MiM_WRITE_OFFSET]; 										// = read_offset_ + read_cols_*read_rows for linear MipMap.
	
	uint write_rows_	= mipmap_params[MiM_READ_ROWS] /2;
	uint read_cols_ 	= mipmap_params[MiM_READ_COLS];
	uint write_cols_ 	= mipmap_params[MiM_WRITE_COLS];
	
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];   													// whole mipmap
	
	uint write_row   	= global_id_u / write_cols_ ;
	uint write_column 	= fmod(global_id_flt, write_cols_);
	
	uint read_row    	= 2*write_row;
	uint read_column 	= 2*write_column;
	
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column  ;					// NB 4 channels.  + margin
	uint write_index 	= write_offset_ +  write_row * mm_cols  + write_column ;					// write_cols_, use read_cols_ as multiplier to preserve images  + margin
	
	for (int i=0; i<3; i++){																		// Load local_img_patch
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
	barrier(CLK_LOCAL_MEM_FENCE);																	// No 'if->return' before fence between write & read local mem
	float4 reduced_pixel = 0;
	for (int i=0; i<3; i++){
		for (int y=0; y<3; y++){
			reduced_pixel += local_img_patch[lid+y + i*patch_length]/9;								// 3x3 box filter, rather than Gaussian
		}
	}
	if (write_row>write_rows_) return;
	//if (global_id_u >= mipmap_params[MiM_PIXELS]) return;											// num pixels to be written & num threads to really use.
	//reduced_pixel[2] = global_id_flt/(float)(mipmap_params[MiM_PIXELS]); // debugging 
	reduced_pixel[3] = 1.0f;
	img[ write_index] = reduced_pixel;
}


__kernel void  img_grad(
	__constant	uint*		mipmap_params,	//0
	__constant 	uint*		uint_params,	//1
	__constant 	float*		fp32_params,	//2
	__global 	float4*		img,			//3 
	__global 	float4*		gxp,			//4
	__global 	float4*		gyp,			//5
	__global 	float4*		g1p,			//6
	__constant 	float2*		SE3_map,		//7
	__global 	float4*		SE3_grad_map	//8  // We keep hsv sepate at this stage, so 6*4=24, but float16 is the largest type. 
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	
	uint read_offset_ 	= mipmap_params[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params[MiM_READ_ROWS];
	
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint mm_pixels		= uint_params[MM_PIXELS];
	
	uint read_row    	= global_id_u / read_cols_;
	uint read_column 	= fmod(global_id_flt, read_cols_);
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column ;						// NB 4 channels.  + margin
	
	/// adapted
	int upoff			= -(read_row  != 0)*mm_cols;												// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-1) * mm_cols;
    int lfoff			= -(read_column != 0);
	int rtoff			=  (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;
	
	float alphaG		= fp32_params[ALPHA_G];
	float betaG 		= fp32_params[BETA_G];
	 
	float4 pu, pd, pl, pr;
	pr =  img[offset + rtoff];
	pl =  img[offset + lfoff];
	pu =  img[offset + upoff];
	pd =  img[offset + dnoff];

	float4 gx	= { (pr.x - pl.x), (pr.y - pl.y), (pr.z - pl.z), 1.0f }; //{ fabs(pr.x - pl.x), fabs(pr.y - pl.y), fabs(pr.z - pl.z), 1.0f };		// taking the absolute loses the direction of the gradient. 
	float4 gy	= { (pd.x - pu.x), (pd.y - pu.y), (pd.z - pu.z), 1.0f };
	 
	float4 g1  = { \
		 exp(-alphaG * pow(sqrt(gx.x*gx.x + gy.x*gy.x), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.y*gx.y + gy.y*gy.y), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.z*gx.z + gy.z*gy.z), betaG) ), \
		 1.0f };
	if (global_id_u >= mipmap_params[MiM_PIXELS]) return;	 
	g1p[offset]= g1;
	gxp[offset]= fabs(gx);
	gyp[offset]= fabs(gy);
	
	for (uint i=0; i<6; i++) {	
		float2 SE3_px =  SE3_map[read_index + i* mm_pixels];										// SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;   float2 partial_gradient={u_flt-u2 , v_flt-v2}; // Find movement of pixel
		float4 SE3_grad_px = (gx*SE3_px[0])  +  (gy*SE3_px[1]);
		SE3_grad_px[3] = 1;
		SE3_grad_map[read_index + i* mm_pixels]  =  SE3_grad_px; 
		
		if (global_id_u ==1)printf("\nSE3_grad_map[%u + %u * %u] = (%f, %f, %f, %f),  SE3_px=(%f,%f),  gx=(%f, %f, %f, %f),  gy=(%f, %f, %f, %f)" \
		, read_index, i, mm_pixels,   SE3_grad_px[0], SE3_grad_px[1], SE3_grad_px[2], SE3_grad_px[3],    SE3_px[0], SE3_px[1]  \
		, gx[0], gx[1], gx[2], gx[3],   gy[0], gy[1], gy[2], gy[3] \
				  );
	} 
}


__kernel void compute_param_maps(
	__constant 	uint*	mipmap_params,	//0
	__constant 	uint*	uint_params,	//1
	__constant 	float* 	SO3_k2k,		//2
	__global 	float2*	SE3_map			//3
	
	//__constant 	float*	fp32_params,//1
	//__global 	float* 	depth_map,		//4
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
	float u_flt			= u * reduction;							// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	
	//if (global_id_u == 0 || global_id_u == mipmap_params[MiM_PIXELS]-1 ) printf("\nreduction=%u,  read_offset_=%u  , read_cols_=%u,  mipmap_params[MiM_PIXELS]=%u, global_id_u=%u,   u=%u,  v=%u, u_flt=%f, v_flt=%f,  uint_params[MM_PIXELS]=%u,  ", \
	//	reduction, mipmap_params[MiM_READ_OFFSET], mipmap_params[MiM_READ_COLS], mipmap_params[MiM_PIXELS], global_id_u, u, v, u_flt, v_flt, uint_params[MM_PIXELS]  );
	
	for (uint i=0; i<6; i++) {										// for each SE3 DoF
		// Find new pixel position, h=homogeneous coords.
		int idx = i *16;
		float inv_depth = 1.0f;// mid point max-min inv depth
		float uh2 = SO3_k2k[idx+0]*u_flt + SO3_k2k[idx+1]*v_flt + SO3_k2k[idx+2]*1 + SO3_k2k[idx+3]*inv_depth;
		float vh2 = SO3_k2k[idx+4]*u_flt + SO3_k2k[idx+5]*v_flt + SO3_k2k[idx+6]*1 + SO3_k2k[idx+7]*inv_depth;
		float wh2 = SO3_k2k[idx+8]*u_flt + SO3_k2k[idx+9]*v_flt + SO3_k2k[idx+10]*1+ SO3_k2k[idx+11]*inv_depth;
		//float h/z  = SO3_k2k[12]*u_flt + SO3_k2k[13]*v + SO3_k2k[14]*1; // +SO3_k2k[15]/z
	
		float u2   = uh2/wh2;
		float v2   = vh2/wh2;
		float2 partial_gradient={u_flt-u2 , v_flt-v2}; // Find movement of pixel
		
		// (global_id_u == 10 || global_id_u == 11)
		//if  (u==0 && v <5) printf("\n partial_gradient= { %f - %f = %f,  %f - %f = %f }, i=%u,     uint_params[MM_PIXELS]=%u,    (read_index + i* uint_params[MM_PIXELS])=%u     ",  \
		//	 u_flt,  u2, (u_flt-u2), v_flt, v2, (v_flt-v2), i, uint_params[MM_PIXELS], read_index+i*uint_params[MM_PIXELS]  );
		
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

__kernel void se3_grad(
	__constant 	uint*	mipmap_params,			//0
	__constant 	uint*	uint_params,			//1
	__constant  float*  fp32_params,			//2
	__global	float16*k2k,					//3
	__global 	float4*	img_cur,				//4 
	__global 	float4*	img_new,				//5
	__global 	float4*	SE3_grad_map_cur_frame,	//6
	__global 	float4*	SE3_grad_map_new_frame,	//7
	__global	float* 	inv_depth_map,			//8
	__local		float8*	local_sum_grads,		//9
	__global	float8*	global_sum_grads,		//10
	__global 	float4*	SE3_incr_map_			//11
	)
{// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	// if (global_id_u >= mipmap_params[MiM_PIXELS]) { local_sum_grads[lid]=0;   return;}	// zero unitialized local_sum_grads;
	
	uint local_size 	= get_local_size(0);
	uint group_size 	= local_size;
	uint work_dim 		= get_work_dim();
	uint global_size	= get_global_size(0);
	float16 k2k_pvt		= k2k[0];
	
	uint read_offset_ 	= mipmap_params[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params[MiM_READ_ROWS];
	
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint mm_pixels		= uint_params[MM_PIXELS];
	
	float SE3_LM_a		= fp32_params[SE3_LM_A];											// Optimisation parameters
	float SE3_LM_b		= fp32_params[SE3_LM_B];
	
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;											// read_row
	uint u 				= fmod(global_id_flt, read_cols_);									// read_column
	float u_flt			= u * reduction;													// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	
	float inv_depth = inv_depth_map[read_index]; 											//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.
	float uh2 = k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1 + k2k_pvt[3]*inv_depth;
	float vh2 = k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1 + k2k_pvt[7]*inv_depth;
	float wh2 = k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1+ k2k_pvt[11]*inv_depth;
	//float h/z  = k2k_pvt[12]*u_flt + k2k_pvt[13]*v + k2k_pvt[14]*1; // +k2k_pvt[15]/z
	
	float u2_flt	= uh2/wh2;
	float v2_flt	= vh2/wh2;
	int  u2			= floor((u2_flt/reduction)+0.5f) ;										// nearest neighbour interpolation
	int  v2			= floor((v2_flt/reduction)+0.5f) ;										// NB this corrects the sparse sampling to the redued scales.
	uint read_index_new = read_offset_ + v2*read_cols_ + u2;
	
	float8 grads= 0;
	grads[6]=1;
	if ((u2>read_cols_/2) && (u2<(10+(read_cols_/2)))  && (v2==0)) printf("\nread_cols_=%u,  read_rows_=%u, read_offset_=%u,  u=%u, v=%u, u2=%u, v2=%u,  reduction=%u,  u_flt=%f,   v_flt=%f", read_cols_, read_rows_, read_offset_, u, v, u2, v2, reduction, u_flt, v_flt);
	
	if (  !((u2<0) || (u2>=read_cols_) || (v2<0) || (v2>=read_rows_))  ) {					// if (not within new frame) skip  Problem u2&v2 are wrong.
		int idx = 0;
		float4 rho = img_cur[read_index] - img_new[read_index_new];
		for (uint i=0; i<6; i++) {															// for each SE3 DoF
			float4 SE3_grad_cur_px = SE3_grad_map_cur_frame[read_index     + i * mm_pixels ] ;
			float4 SE3_grad_new_px = SE3_grad_map_new_frame[read_index_new + i * mm_pixels ] ;
			float4 delta = {0.0f,0.0f,0.0f,1.0f};
			for (uint j=0; j<3; j++){
				float SE3_grad = (SE3_grad_cur_px[j] );//+ SE3_grad_new_px[j]);				// Using mean of jacobiand produces lots of noise. 
				if (fabs(SE3_grad)> 0.01 ) delta[j] = (-2 * rho[j] / SE3_grad) ;			// Descent on the mean of the gradient : image intensity / SE3 , protected against zero gradient null regions of SE3.
			}
			grads[i] = delta[0] + delta[1] + delta[2];
			SE3_incr_map_[read_index + i * mm_pixels ] = delta;
			if (global_id_u==1){
				printf("\n\ni=%u,  \nread_index=%u, \nread_index_new=%u, \nSE3_LM_a=%f , \nSE3_LM_b=%f,   \nrho=(%f,%f,%f,%f), \nSE3_grad_cur_px=(%f,%f,%f,%f), \nSE3_grad_new_px=(%f,%f,%f,%f), \ndelta=(%f,%f,%f,%f)"\
				,i, read_index, read_index_new, SE3_LM_a, SE3_LM_b \
				,rho[0] ,rho[1] ,rho[2] ,rho[3] \
				,SE3_grad_cur_px[0],SE3_grad_cur_px[1],SE3_grad_cur_px[2],SE3_grad_cur_px[3]\
				,SE3_grad_new_px[0],SE3_grad_new_px[1],SE3_grad_new_px[2],SE3_grad_new_px[3]\
				,delta[0], delta[1], delta[2], delta[3]\
				); 
			}
		}
		grads[7]=1;																			// Count hits, and divide group by num hits, without using atomics!
	}
	local_sum_grads[lid] = grads  ;
	////////////////////////////////////////////////////////////////////////////////////////// Reduction
	int max_iter = ilogb((float)(group_size));
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)		// problem : how to produce one result for each mipmap layer ?  NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);														// No 'if->return' before fence between write & read local mem
		if (lid<group_size)  local_sum_grads[lid] += local_sum_grads[lid+group_size];		// local_sum_grads  
	}
	if (lid==0) {
		uint group_id 	= get_group_id(0);
		uint global_sum_offset = read_offset_ / local_size ;								// Compute offset for this layer
		uint num_groups = get_num_groups(0);
		
		printf("\n\n reduction=%u,  global_sum_offset=%u,  num_groups=%u,  group_id=%u, read_index=%u, grads[6]=%f, grads[7]=%f, \nlocal_sum_grads[lid]=( %f, %f, %f, %f,   %f, %f, %f, %f ),  \nglobal_sum_grads[group_id]=( %f, %f, %f, %f,   %f, %f, %f, %f ) "\
		, reduction, global_sum_offset,  num_groups, group_id, read_index, grads[6], grads[7]\
		, local_sum_grads[lid][0],local_sum_grads[lid][1],local_sum_grads[lid][2],local_sum_grads[lid][3], local_sum_grads[lid][4],local_sum_grads[lid][5],local_sum_grads[lid][6],local_sum_grads[lid][7] \
		, global_sum_grads[group_id][0], global_sum_grads[group_id][1], global_sum_grads[group_id][2], global_sum_grads[group_id][3],    global_sum_grads[group_id][4], global_sum_grads[group_id][5], global_sum_grads[group_id][6], global_sum_grads[group_id][7]\
		);
		
		float8 layer_data = {num_groups, reduction, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };	// Write layer data to first entry
		if (global_id_u == 0) {global_sum_grads[global_sum_offset] = layer_data; }
		global_sum_offset += 1+ group_id;
		
		if (local_sum_grads[0][7] >0){
			global_sum_grads[global_sum_offset] = local_sum_grads[0] / local_sum_grads[0][7];// Save to global_sum_grads // Count hits, and divide group by num hits, without using atomics!
		}else global_sum_grads[global_sum_offset] = 0;
	}
}

__kernel void reduce (
	__constant 	uint*		mipmap_params,	//0
	__constant 	uint*		uint_params,	//1
	__global	float8*		se3_sum,		//2
	__local		float8*		local_sum_grads,//3
	__global 	float8*		se3_sum2		//4
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
	
	uint global_sum_offset 	= read_offset_ / local_size ;								// Compute offset for this layer
	//
	local_sum_grads[lid] = se3_sum[global_sum_offset  + global_id_u];
	int max_iter = ilogb((float)(group_size));
	
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)	// problem : how to produce one result for each mipmap layer ?  NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);													// No 'if->return' before fence between write & read local mem
		if (lid<group_size)  local_sum_grads[lid] += local_sum_grads[lid+group_size];	// local_sum_grads  
	}
	
	if (lid==0) {
		uint group_id 	= get_group_id(0);
		uint global_sum_offset = read_offset_ / local_size ;								// Compute offset for this layer
		uint num_groups = get_num_groups(0);
		
		printf("\n\n reduction=%u,  global_sum_offset=%u,  num_groups=%u,  group_id=%u, \nlocal_sum_grads[lid]=( %f, %f, %f, %f,   %f, %f, %f, %f ),  \nse3_sum2[group_id]=( %f, %f, %f, %f,   %f, %f, %f, %f ) "\
		, reduction, global_sum_offset,  num_groups, group_id \
		, local_sum_grads[lid][0],local_sum_grads[lid][1],local_sum_grads[lid][2],local_sum_grads[lid][3], local_sum_grads[lid][4],local_sum_grads[lid][5],local_sum_grads[lid][6],local_sum_grads[lid][7] \
		, se3_sum2[group_id][0], se3_sum2[group_id][1], se3_sum2[group_id][2], se3_sum2[group_id][3],    se3_sum2[group_id][4], se3_sum2[group_id][5], se3_sum2[group_id][6], se3_sum2[group_id][7]\
		);
		
		float8 layer_data = {num_groups, reduction, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };	// Write layer data to first entry
		if (global_id_u == 0) {se3_sum2[global_sum_offset] = layer_data; }
		global_sum_offset += 1+ group_id;
		
		if (local_sum_grads[0][7] >0){
			se3_sum2[global_sum_offset] = local_sum_grads[0] / local_sum_grads[0][7];// Save to se3_sum2 // Count hits, and divide group by num hits, without using atomics!
		}else se3_sum2[global_sum_offset] = 0;
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

