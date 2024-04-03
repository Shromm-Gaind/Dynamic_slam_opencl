#include "kernels_macros.h"
#include "kernels.h"

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
	uint8 mipmap_params_= mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
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

		SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;
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
	// inputs:
	__private	uint	layer,					//0
	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3
	__global	float*  so3_k2k,				//4
	__global 	float4*	img_cur,				//5
	__global 	float4*	img_new,				//6
	__global 	float8*	SE3_grad_map_cur_frame,	//7
	__global 	float8*	SE3_grad_map_new_frame,	//8
	// outputs:
	__local		float4*	local_sum_grads,		//9
	__global	float4*	global_sum_grads,		//10
	__global 	float4*	SE3_incr_map_,			//11
	__global	float4* Rho_,					//12
	__local		float4*	local_sum_rho_sq,		//13												// 1 DoF, float4 channels
	__global 	float4*	global_sum_rho_sq		//14
	)
 {																									// find gradient wrt SE3 find global sum for each of the 6 DoF ##################################################
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);

	uint local_size 	= get_local_size(0);
	uint group_size 	= local_size;
	uint work_dim 		= get_work_dim();
	uint global_size	= get_global_size(0);

	uint8 mipmap_params_= mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];											// Width of this layer of the image pyramid.
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels	= mipmap_params_[MiM_PIXELS];

	uint base_cols		= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];														// Width of the whole buffer.
	uint mm_pixels		= uint_params[MM_PIXELS];

	float SE3_LM_a		= fp32_params[SE3_LM_A];													// Optimisation parameters
	float SE3_LM_b		= fp32_params[SE3_LM_B];

	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;													// read_row
	uint u 				= fmod(global_id_flt, read_cols_);											// read_column
	float u_flt			= u * reduction;															// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;										// Where to read keyframe #######################################################################################

	float uh2 = so3_k2k[0]*u_flt + so3_k2k[1]*v_flt + so3_k2k[2]*1;
	float vh2 = so3_k2k[4]*u_flt + so3_k2k[5]*v_flt + so3_k2k[6]*1;
	float wh2 = so3_k2k[8]*u_flt + so3_k2k[9]*v_flt + so3_k2k[10]*1;

	float u2_flt	= uh2/wh2;
	float v2_flt	= vh2/wh2;
	int  u2			= floor((u2_flt/reduction)+0.5f) ;												// nearest neighbour interpolation
	int  v2			= floor((v2_flt/reduction)+0.5f) ;												// NB this corrects the sparse sampling to the redued scales.
	/*
									if(u==10 && v==10){
										float so3[16];
										for(int i=0; i<16; i++){so3[i] =  so3_k2k[i];}
										printf("\n__kernel void so3_grad(..) u=10, v=10, u2=%i, v2=%i, reduction=%u,  so3_k2k[]=(%f, %f, %f, %f,    %f, %f, %f, %f,    %f, %f, %f, %f,    %f, %f, %f, %f),"
										, u2, v2, reduction, so3[0], so3[1], so3[2], so3[3], so3[4], so3[5], so3[6], so3[7], so3[8], so3[9], so3[10], so3[11], so3[12], so3[13], so3[14], so3[15] );
									}
	*/
	uint read_index_new = read_offset_ + v2 * mm_cols  + u2; 										// Where to read new frame ######################################################################################
	uint num_DoFs = 3;
	float4 rho = {0.0f,0.0f,0.0f,0.0f}, zero_f4={0.0f,0.0f,0.0f,0.0f};
	float intersection = (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2>2) && (u2<=read_cols_-2) && (v2>2) && (v2<=read_rows_-2)  &&  (global_id_u<=layer_pixels);
																									// chk (u,v,u2,v2) are within bounds of image. ##########
	if (!intersection) read_index_new = 0;															// Prevent out of bounds reading of buffer.

	for (int i=0; i<num_DoFs; i++) 		local_sum_grads[i*local_size + lid] 	= 0;				// Zero the local mem.
	rho     				= intersection * ( img_cur[read_index] - img_new[read_index_new] ) ;	// Comute photometric error (HSV) ##### zero if !intersection  ##################################################
	rho.w					= intersection;															// Set transparency for Rho map.
	Rho_[read_index]      	= rho;																	// save pixelwise photometric error map to buffer. NB Outside if(){}, to zero non-overlapping pixels.
	float4 rho_sq         	= {rho.x*rho.x,  rho.y*rho.y,  rho.z*rho.z, intersection};				// Intersection => count valid pixels.
	local_sum_rho_sq[lid] 	= rho_sq;																// Also compute global Rho^2.


	for (uint i=0; i<num_DoFs; i++) {																// For each SO3 DoF compute "delta" increment for each SO3 Dof, according to each colour channel ################
		float4 delta4 								= zero_f4;
		float8 SE3_grad_cur_px 						= SE3_grad_map_cur_frame[read_index     + i * mm_pixels ] ;
		float8 SE3_grad_new_px 						= SE3_grad_map_new_frame[read_index_new + i * mm_pixels ] ;
		for (int j=0; j<3; j++)		delta4[j] 		= rho[j] * (SE3_grad_cur_px[j] + SE3_grad_cur_px[j+4] + SE3_grad_new_px[j] + SE3_grad_new_px[j+4]); // Photometic error * Sum for each channel of row & col SE3 img gradients of cur and new frame.
		delta4.w									= intersection;									// delta4.w = 1 if pixel is valid, else 0
		local_sum_grads[i*local_size + lid] 		= delta4;										// write grads to local mem for summing over the work group.
		SE3_incr_map_[read_index + i * mm_pixels ]	= delta4;
	}
	for (uint i=3; i<6; i++)	SE3_incr_map_[read_index + i * mm_pixels ]		= zero_f4;			// zero the unused SE3 translation DoFs, in SE3 rotation only.

	////////////////////////////////////////////////////////////////////////////////////////		// Reduction : sum delta4 & rho_sq for this work group ##########################################################
	int max_iter = 9;										// ceil( log2( (float)(group_size) ) );
	for (uint iter=0; iter<=max_iter ; iter++) {			// for log2(local work group size)		// problem : how to produce one result for each mipmap layer ?
																									// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size){
			for (int i=0; i<num_DoFs; i++){
				local_sum_grads[i*local_size + lid] 			+= local_sum_grads[i*local_size + lid + group_size];	// local_sum_grads
			}
			local_sum_rho_sq[lid] 								+= local_sum_rho_sq[lid + group_size];					// Also compute global Rho^2.
		}
	}																													// NB Now local_sum_grads[0].w holds count of valid pixels.

	barrier(CLK_LOCAL_MEM_FENCE);																						// Wite to buffer : sum delta4 & rho_sq for this work group #################################
	if (lid==0) {
		uint group_id 											= get_group_id(0);
		uint rho_global_sum_offset 								= read_offset_ / local_size ;							// Compute offset for this layer ###############
		uint se3_global_sum_offset 								= rho_global_sum_offset *num_DoFs;						// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
		uint num_groups 										= get_num_groups(0);

		float4 layer_data 										= {num_groups,  reduction,  0.0f,  0.0f };				// Write layer data to first entry #############
		if (global_id_u == 0) {
			global_sum_rho_sq[rho_global_sum_offset] 			= layer_data;
			global_sum_grads[se3_global_sum_offset] 			= layer_data;
		}
		rho_global_sum_offset 									+= 1 + group_id;
		se3_global_sum_offset 									+= (1 + group_id) * num_DoFs;

		global_sum_rho_sq[rho_global_sum_offset] 				= local_sum_rho_sq[lid];								// global_sum_rho_sq[rho_global_sum_offset].w  holds num valid pixels.
		for (int i=0; i<num_DoFs; i++){																					// for SO3 DoFs
			float4 temp_float4 									= local_sum_grads[i*local_size + lid] / local_size  ;	// Divide by num pixels in the work group. This prevents partly emty groups from being over weighted.
			global_sum_grads[se3_global_sum_offset + i] 		= temp_float4 ;											// global_sum_grads[se3_global_sum_offset + i].w holds : (vaild pixels)/(num threads in group).
		}
	}
}

__kernel void se3_Rho_sq(
	// inputs
	__private	uint	layer,					//0
	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3
	__global	float16*k2k,					//4		// keyframe2K
	__global 	float4*	img_cur,				//5		// keyframe
	__global 	float4*	img_new,				//6
	__global	float* 	depth_map,				//7		// NB keyframe GT_depth, now stored as inv_depth
	// outputs
	__global	float4* Rho_,					//8
	__local		float4*	local_sum_rho_sq,		//9		// 1 DoF, float4 channels
	__global 	float4*	global_sum_rho_sq		//10
	)
 {																									// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint  global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint  lid 			= get_local_id(0);

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

	float inv_depth 	= depth_map[read_index ]; 									//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.//inv dept  //depth_index
	float uh2 			= k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1 + k2k_pvt[3]*inv_depth;
	float vh2 			= k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1 + k2k_pvt[7]*inv_depth;
	float wh2 			= k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1+ k2k_pvt[11]*inv_depth;
	//float h/z  		= k2k_pvt[12]*u_flt + k2k_pvt[13]*v + k2k_pvt[14]*1; // +k2k_pvt[15]/z

	float u2_flt		= uh2/wh2;
	float v2_flt		= vh2/wh2;
	int  u2				= floor((u2_flt/reduction)+0.5f) ;											// nearest neighbour interpolation
	int  v2				= floor((v2_flt/reduction)+0.5f) ;											// NB this corrects the sparse sampling to the redued scales.
	uint read_index_new = read_offset_ + v2 * mm_cols  + u2; // read_cols_
	uint num_DoFs 		= 6;
	float4 new_px;

	local_sum_rho_sq[lid] = 0;																		// Essential to zero local mem.
/*
	if (global_id_u==1){
		printf("\nkernel se3_Rho_sq(..): u=%i,  v=%i,   inv_depth=%f, u2=%f,  v2=%f,  u2_flt=%f,  v2_flt=%f,    k2k_pvt=(%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f)"\
		,u, v, inv_depth, u_flt, v_flt, u2_flt, v2_flt,  k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],   k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],   k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15]   )  ;
	}
*/
	float4 rho 											= {0.0f,0.0f,0.0f,0.0f};
																									// Exclude all out-of-bounds threads:
	float intersection = (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2>2) && (u2<=read_cols_-2) && (v2>2) && (v2<=read_rows_-2)  &&  (global_id_u<=layer_pixels);

	if (  intersection  ) {																			// if (not cleanly within new frame) skip  Problem u2&v2 are wrong.
		int idx 										= 0;										// float4 bilinear_flt4(__global float4* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);
		new_px 											= bilinear_flt4(img_new, u2_flt/reduction, v2_flt/reduction, mm_cols, read_offset_);
		rho 											= img_cur[read_index] - new_px;
		rho[3] 											= alpha;

		Rho_[read_index] 								= rho;										// save pixelwise photometric error map to buffer. NB Outside if(){}, to zero non-overlapping pixels.
		float4 rho_sq 									= {rho.x*rho.x,  rho.y*rho.y,  rho.z*rho.z, rho.w};
		local_sum_rho_sq[lid] 							= rho_sq;									// Also compute global Rho^2.

		//if (layer==5) printf(",(%u,%f)", global_id_u ,inv_depth);									// debug chk on value of inv_depth
	}
	////////////////////////////////////////////////////////////////////////////////////////		// Reduction
	int max_iter = 9;//ceil(log2((float)(group_size)));
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?
																									// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   			/= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size){
			local_sum_rho_sq[lid] += local_sum_rho_sq[lid + group_size];							// Also compute global Rho^2.
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid==0) {
		uint group_id 									= get_group_id(0);
		uint rho_global_sum_offset 						= read_offset_ / local_size ;				// Compute offset for this layer
		uint num_groups 								= get_num_groups(0);
		//printf("\nse3_Rho_sq(..): layer=%i, u=%i, v=%i, group_id=%i,  rho_global_sum_offset=%i,  (float)read_offset_/local_size=%f,  local_size=%u, read_offset_=%u,  local_sum_rho_sq[lid]=(%f,%f,%f,%f), local_sum_rho_sq[lid][3]=%f ",
		//	   layer, u, v, group_id, rho_global_sum_offset, ((float)read_offset_)/((float)local_size),  local_size, read_offset_, local_sum_rho_sq[lid].x, local_sum_rho_sq[lid].y, local_sum_rho_sq[lid].z, local_sum_rho_sq[lid].w, local_sum_rho_sq[lid][3] );


		float4 layer_data 								= {num_groups, reduction, 0.0f, 0.0f };		// Write layer data to first entry
		if (global_id_u == 0) {
			global_sum_rho_sq[rho_global_sum_offset] 	= layer_data;
		}
		rho_global_sum_offset 							+= 1 + group_id;

		if (local_sum_rho_sq[lid][3] >0){															// Using last channel rho[3], to count valid pixels being summed.
			global_sum_rho_sq[rho_global_sum_offset] 	= local_sum_rho_sq[lid];
			//printf("\nkernel se3_Rho_sq(..)_2: layer=%i,  group_id=%i,   local_sum_rho_sq[lid]=(%f,%f,%f,%f )", layer, group_id,  local_sum_rho_sq[lid].x, local_sum_rho_sq[lid].y, local_sum_rho_sq[lid].z, local_sum_rho_sq[lid].w );
		}else {																						// If no matching pixels in this group, set values to zero.
			global_sum_rho_sq[rho_global_sum_offset] 	= 0;
		}
	}
}
///////////////////////////////////////////////////////

//	4 channel se3_grad
__kernel void se3_grad(
	// inputs
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
	__global	float4* Rho_,					//10
	// outputs
	__local		float4*	local_sum_grads,		//11
	__global	float4*	global_sum_grads,		//12
	__global 	float4*	SE3_incr_map_			//13
/*
	//__local		float4*	local_sum_rho_sq,		//14	1 DoF, float4 channels
	//__global 	float4*	global_sum_rho_sq			//15
*/
	)
 {																									// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint  global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint  lid 			= get_local_id(0);

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

	float inv_depth 	= depth_map[read_index]; 									//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.//inv dept  //depth_index
	float uh2 			= k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1 + k2k_pvt[3]*inv_depth;
	float vh2 			= k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1 + k2k_pvt[7]*inv_depth;
	float wh2 			= k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1+ k2k_pvt[11]*inv_depth;
	//float h/z  		= k2k_pvt[12]*u_flt + k2k_pvt[13]*v + k2k_pvt[14]*1; // +k2k_pvt[15]/z

	float u2_flt		= uh2/wh2;
	float v2_flt		= vh2/wh2;
	int  u2				= floor((u2_flt/reduction)+0.5f) ;												// nearest neighbour interpolation
	int  v2				= floor((v2_flt/reduction)+0.5f) ;												// NB this corrects the sparse sampling to the redued scales.
	uint read_index_new = read_offset_ + v2 * mm_cols  + u2; // read_cols_
	uint num_DoFs 		= 6;
	float4 new_px;
/*
	if (global_id_u==1){
		printf("\nkernel se3_grad_a(..): layer=%i,  u=%i,  v=%i,   inv_depth=%f, u2=%f,  v2=%f,  u2_flt=%f,  v2_flt=%f,    k2k_pvt=(%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f)"\
		,layer ,u, v, inv_depth, u_flt, v_flt, u2_flt, v2_flt,  k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],   k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],   k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15]   )  ;
	}
*/
	float4 rho 													= Rho_[read_index]; // {0.0f,0.0f,0.0f,0.0f};
/*
	//if (lid==0){printf("\nkernel se3_grad_b(..): layer=%i,  u=%i,  v=%i,  rho=(%f,%f,%f,%f)",layer , u, v, rho.x, rho.y, rho.z, rho.w  );}
*/
	for (int i=0; i<6; i++) local_sum_grads[i*local_size + lid] = 0;								// Essential to zero local mem.
																									// Exclude all out-of-bounds threads:
	float intersection 											= rho.w;	//= (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2>2) && (u2<=read_cols_-2) && (v2>2) && (v2<=read_rows_-2)  &&  (global_id_u<=layer_pixels);

	if (  intersection  ) {																			// if (not cleanly within new frame) skip  Problem u2&v2 are wrong.
		int idx 												= 0;								// float4 bilinear_flt4(__global float4* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);
/*
		//new_px 												= bilinear_flt4(img_new, u2_flt/reduction, v2_flt/reduction, mm_cols, read_offset_);
		//rho 													= img_cur[read_index] - new_px;
		//rho[3] 												= alpha;

		//Rho_[read_index] 										= rho;								// save pixelwise photometric error map to buffer. NB Outside if(){}, to zero non-overlapping pixels.
		//float4 rho_sq 											= {rho.x*rho.x,  rho.y*rho.y,  rho.z*rho.z, rho.w};
		//local_sum_rho_sq[lid] 									= rho_sq;							// Also compute global Rho^2.
*/
		for (uint i=0; i<6; i++) {																	// for each SE3 DoF
			float8 SE3_grad_cur_px 								= SE3_grad_map_cur_frame[read_index     + i * mm_pixels ] ;
			float8 SE3_grad_new_px 								= bilinear_SE3_grad (SE3_grad_map_new_frame, u2_flt, v2_flt, mm_cols, read_offset_);	// SE3_grad_map_new_frame[read_index_new + i * mm_pixels ] ;

			float4 delta4;
			delta4.w											= alpha;
			for (int j=0; j<3; j++) {
				float SE3_grad =  (SE3_grad_cur_px[j] + SE3_grad_cur_px[j+4] + SE3_grad_new_px[j] + SE3_grad_new_px[j+4]);
				delta4[j] = 0;
				float thresh = 0.01;
				if (SE3_grad > thresh)  delta4[j] 				= rho[j] / SE3_grad;				// Take step large enough to correct Rho, if there is non-zero gradient.
			}
			local_sum_grads[i*local_size + lid] 				= delta4;							// write grads to local mem for summing over the work group.
			SE3_incr_map_[read_index + i * mm_pixels ] 			= delta4;
		}
		for (uint i=0; i<3; i++) {	// translation, amplify nearby movement.						// NB SE3_incr_map_[ ].w = alpha for the image within the mipmap.
			SE3_incr_map_[read_index + i * mm_pixels ].x 		*= inv_depth * 100;
			SE3_incr_map_[read_index + i * mm_pixels ].y 		*= inv_depth * 100;
			SE3_incr_map_[read_index + i * mm_pixels ].z 		*= inv_depth * 100;
		}

		for (uint i=3; i<6; i++) {	// rotation, amplify distant movement.							// NB SE3_incr_map_[ ].w = alpha for the image within the mipmap.
			SE3_incr_map_[read_index + i * mm_pixels ].x 		/= (inv_depth * 50);
			SE3_incr_map_[read_index + i * mm_pixels ].y 		/= (inv_depth * 50);
			SE3_incr_map_[read_index + i * mm_pixels ].z 		/= (inv_depth * 50);
		}

		//if (layer==5) printf(",(%u,%f)", global_id_u ,inv_depth);									// debug chk on value of inv_depth
	}
	////////////////////////////////////////////////////////////////////////////////////////		// Reduction
	int max_iter 												= 9;								//ceil(log2((float)(group_size)));
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?
																									// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   											/= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size){
			for (int i=0; i<num_DoFs; i++){															// local_sum_grads
				local_sum_grads[i*local_size + lid] 			+= local_sum_grads[i*local_size + lid + group_size];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid==0) {
		uint group_id 											= get_group_id(0);
		//uint se3_global_sum_offset 								= read_offset_ *num_DoFs / local_size ;		// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
		uint rho_global_sum_offset 								= read_offset_ / local_size ;					// Compute offset for this layer,   NB interger rounding !
		uint se3_global_sum_offset 								= rho_global_sum_offset *num_DoFs;				// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
		//printf("\nkernel se3_grad_c(..): layer=%i,  group_id=%i,  read_offset_/local_size=%f,  local_size=%u, read_offset_=%u ", layer, group_id, ((float)read_offset_)/((float)local_size),  local_size, read_offset_);

		if (global_id_u == 0) {
			uint num_groups 									= get_num_groups(0);
			float4 layer_data 									= {num_groups, reduction, 0.0f, 0.0f };			// Write layer data to first entry
			global_sum_grads[se3_global_sum_offset] 			= layer_data;

			//printf("\nkernel se3_grad_d(..)_2: layer=%i,  group_id=%i,  se3_global_sum_offset=%i,  layer_data=(%f,%f,%f,%f),    u=%i,  v=%i,   inv_depth=%f, u2=%f,  v2=%f,  u2_flt=%f,  v2_flt=%f,    k2k_pvt=(%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f),\t rho=(%f,%f,%f,%f), local_sum_grads=(%f,%f,%f,%f)"\
			//	,layer, group_id, se3_global_sum_offset,  layer_data.x, layer_data.y, layer_data.z, layer_data.w,  u, v, inv_depth, u_flt, v_flt, u2_flt, v2_flt,  k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],   k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],   k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15], rho.x, rho.y, rho.z, rho.w, local_sum_grads[0][0],local_sum_grads[0][1],local_sum_grads[0][2],local_sum_grads[0][3]   )  ;
		}

		se3_global_sum_offset 									+= num_DoFs+ group_id*num_DoFs;

		if (local_sum_grads[0][3] >0){																// Using last channel local_sum_pix[0][7], to count valid pixels being summed.
			//global_sum_rho_sq[rho_global_sum_offset]			= local_sum_rho_sq[lid];
			for (int i=0; i<num_DoFs; i++){
				float4 temp_float4 								= local_sum_grads[i*local_size + lid] / local_size; 	//   / local_sum_grads[i*local_size + lid].w ;  Better to divide by local size, preserve information wrt number of valid pixels.
				global_sum_grads[se3_global_sum_offset + i] 	= temp_float4 ;						// local_sum_grads
				//printf("\nkernel se3_grad_d(..)_2: layer=%i,  group_id=%i,   local_sum_grads[i*local_size + lid]=(%f,%f,%f,%f )", layer, group_id,   temp_float4.x,temp_float4.y,temp_float4.z,temp_float4.w );
			}																						// Save to global_sum_grads // Count hits, and divide group by num hits, without using atomics!
		}else {																						// If no matching pixels in this group, set values to zero.
			for (int i=0; i<num_DoFs; i++){
				global_sum_grads[se3_global_sum_offset] 		= 0;	/*[rho_global_sum_offset]*/
			}
		}
	}
}

///////////
__kernel void se3_LK_grad(
	// inputs
	__private	uint	layer,					//0
	__private	uint	channel,				//1 ############
	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3
	__global	float16*k2k,					//4		// keyframe2K
	__global 	float4*	img_cur,				//5		// keyframe
	__global 	float4*	img_new,				//6

	//__global 	float8*	HSV_grad,				//7

	__global 	float8*	SE3_grad_map_cur_frame,	//7		// keyframe
	__global 	float8*	SE3_grad_map_new_frame,	//8
	__global	float* 	depth_map,				//9		// NB keyframe GT_depth, now stored as inv_depth

	// outputs
	__global	float4* Rho_,					//10
	__local		float4*	local_sum_rho_sq,		//11	1 DoF, float 1 channels	NB local memory limited to 4kb on rtx3080 and local worksize is 1024
	__global 	float4*	global_sum_rho_sq,		//12

	__global	float4* weights_map,			//13
	__local		float*	local_sum_weight,		//14	6 DoF, float 1 channels
	__global 	float*	global_sum_weight,		//15

	__global 	float4*	SE3_incr_map_,			//16
	__local		float*	local_sum_grads,		//17	6 DoF, float 1 channels
	__global	float*	global_sum_grads		//18
	)
 {																									// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint  global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint  lid 			= get_local_id(0);
																	if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_1"   ); }
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

	uint reduction		= mm_cols/read_cols_;
	uint v 				= global_id_u / read_cols_;																		// read_row
	uint u 				= fmod(global_id_flt, read_cols_);																// read_column
	float u_flt			= u * reduction;																				// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	float alpha			= img_cur[read_index].w;

	float inv_depth 	= depth_map[read_index]; 																		//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.//inv dept  //depth_index
	float uh2 			= k2k_pvt[0]*u_flt 	+ k2k_pvt[1]*v_flt 	+ k2k_pvt[2]*1 	+ k2k_pvt[3]*inv_depth;
	float vh2 			= k2k_pvt[4]*u_flt 	+ k2k_pvt[5]*v_flt 	+ k2k_pvt[6]*1 	+ k2k_pvt[7]*inv_depth;
	float wh2 			= k2k_pvt[8]*u_flt 	+ k2k_pvt[9]*v_flt 	+ k2k_pvt[10]*1	+ k2k_pvt[11]*inv_depth;
	//float h/z  		= k2k_pvt[12]*u_flt	+ k2k_pvt[13]*v_flt + k2k_pvt[14]*1; // +k2k_pvt[15]/z
	float u2_flt		= uh2/wh2;
	float v2_flt		= vh2/wh2;

	uint num_DoFs 		= 6;
	float4 new_px;
	float4 rho 			= {0.0f,0.0f,0.0f,0.0f};
	float weight;
	float8 se3_incr;
																	if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_2"   ); }

	for (int i=0; i<6; i++) {																							// Essential to zero local mem.
		local_sum_rho_sq[i*local_size + lid] 	= 0;
		local_sum_weight[i*local_size + lid] 	= 0;
		local_sum_grads[i*local_size + lid] 	= 0;
	}
																	if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_3"   ); }

																														// Exclude all out-of-bounds threads:
	float intersection 	= (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2_flt>2) && (u2_flt<=read_cols_-2) && (v2_flt>2) && (v2_flt<=read_rows_-2)  &&  (global_id_u<=layer_pixels);

	if (  intersection  ) {																								// if (not cleanly within new frame) skip  Problem u2&v2 are wrong.
		int idx 					= 0;																				// float4 bilinear_flt4(__global float4* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);
		new_px 						= bilinear_flt4(img_new, u2_flt/reduction, v2_flt/reduction, mm_cols, read_offset_);
		rho 						= img_cur[read_index] - new_px;
		rho[3] 						= alpha;

		Rho_[read_index] 			= rho;																				// save pixelwise photometric error map to buffer. NB Outside if(){}, to zero non-overlapping pixels.
		float4 rho_sq 				= {rho.x*rho.x,  rho.y*rho.y,  rho.z*rho.z, rho.w};
		local_sum_rho_sq[lid] 		= rho_sq;																			// Also compute global Rho^2.

		float weights[6]; // float4
		bilinear_SE3_grad_weight( weights, SE3_grad_map_cur_frame, read_index,  SE3_grad_map_new_frame,  u2_flt,  v2_flt,  mm_cols,  read_offset_,  reduction,  mm_pixels,  alpha, channel );

		for (uint se3_dim=0; se3_dim<6; se3_dim++) 	{
			local_sum_weight[ se3_dim*local_size + lid ]     		=  weights[se3_dim];
		}

		for (uint se3_dim=0; se3_dim<6; se3_dim++) {																	// for each SE3 DoF
			float8 SE3_grad_cur_px 									= SE3_grad_map_cur_frame[read_index     + se3_dim * mm_pixels ] ;
			float8 SE3_grad_new_px 									= bilinear_SE3_grad (SE3_grad_map_new_frame, u2_flt, v2_flt, mm_cols, read_offset_);	// SE3_grad_map_new_frame[read_index_new + se3_dim * mm_pixels ] ;

			//float4 delta4;
			//delta4.w												= alpha;
			float delta;
			//for (int chan=0; chan<3; chan++) {
			int chan = channel;
				float SE3_grad 										= ( SE3_grad_cur_px[chan] + SE3_grad_cur_px[chan+4] + SE3_grad_new_px[chan] + SE3_grad_new_px[chan+4] ) / 4;
				delta		 										= weights[se3_dim] * rho[chan] / SE3_grad;
			//}
			local_sum_grads[se3_dim*local_size + lid] 				= delta;
		}

		for (uint se3_dim=0; se3_dim<3; se3_dim++) {	// translation, amplify nearby movement.						// NB SE3_incr_map_[ ].w = alpha for the image within the mipmap.
			local_sum_grads[se3_dim*local_size + lid] 				*= inv_depth;
			/*
			local_sum_grads[se3_dim*local_size + lid].x 			*= inv_depth;// * 100;
			local_sum_grads[se3_dim*local_size + lid].y 			*= inv_depth;// * 100;
			local_sum_grads[se3_dim*local_size + lid].z 			*= inv_depth;// * 100;
			*/
		}
/*
		for (uint se3_dim=3; se3_dim<6; se3_dim++) {	// rotation, amplify distant movement.							// NB SE3_incr_map_[ ].w = alpha for the image within the mipmap.
			local_sum_grads[se3_dim*local_size + lid].x 			/= (inv_depth * 50);
			local_sum_grads[se3_dim*local_size + lid].y 			/= (inv_depth * 50);
			local_sum_grads[se3_dim*local_size + lid].z 			/= (inv_depth * 50);
		}
*/
		for (uint se3_dim=0; se3_dim<6; se3_dim++) {
			SE3_incr_map_[read_index + se3_dim * mm_pixels ] 		= local_sum_grads[se3_dim*local_size + lid];
		}
	}

																		if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_4"   ); }


	////////////////////////////////////////////////////////////////////////////////////////		// Reduction
	int max_iter 												= 9;								//ceil(log2((float)(group_size)));
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?
																									// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   											/= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size){
			for (int i=0; i<num_DoFs; i++){															// local_sum_grads
				local_sum_rho_sq[i*local_size + lid] 			+= local_sum_rho_sq[i*local_size + lid + group_size];
				local_sum_weight[i*local_size + lid] 			+= local_sum_weight[i*local_size + lid + group_size];
				local_sum_grads[i*local_size + lid] 			+= local_sum_grads[i*local_size + lid + group_size];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
																		if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_5"   ); }


	if (lid==0) {
		uint group_id 											= get_group_id(0);
		uint rho_global_sum_offset 								= read_offset_ / local_size ;					// Compute offset for this layer,   NB interger rounding !
		uint se3_global_sum_offset 								= rho_global_sum_offset *num_DoFs;				// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
																																												//printf("\nkernel se3_grad_c(..): layer=%i,  group_id=%i,  read_offset_/local_size=%f,  local_size=%u, read_offset_=%u ", layer, group_id, ((float)read_offset_)/((float)local_size),  local_size, read_offset_);
		if (global_id_u == 0) {
			uint num_groups 									= get_num_groups(0);
			float4 layer_data 									= {num_groups, reduction, 0.0f, 0.0f };			// Write layer data to first entry
			global_sum_rho_sq [rho_global_sum_offset]			= layer_data;
			global_sum_weight [se3_global_sum_offset] 			= layer_data;
			global_sum_grads  [se3_global_sum_offset] 			= layer_data;

			//printf("\nkernel se3_grad_d(..)_2: layer=%i,  group_id=%i,  se3_global_sum_offset=%i,  layer_data=(%f,%f,%f,%f),    u=%i,  v=%i,   inv_depth=%f, u2=%f,  v2=%f,  u2_flt=%f,  v2_flt=%f,    k2k_pvt=(%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f),\t rho=(%f,%f,%f,%f), local_sum_grads=(%f,%f,%f,%f)"\
			//	,layer, group_id, se3_global_sum_offset,  layer_data.x, layer_data.y, layer_data.z, layer_data.w,  u, v, inv_depth, u_flt, v_flt, u2_flt, v2_flt,  k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],   k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],   k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15], rho.x, rho.y, rho.z, rho.w, local_sum_grads[0][0],local_sum_grads[0][1],local_sum_grads[0][2],local_sum_grads[0][3]   )  ;
		}
		rho_global_sum_offset									+= num_DoFs+ group_id;
		se3_global_sum_offset 									+= num_DoFs+ group_id*num_DoFs;

		if (local_sum_grads[0][3] >0){																// Using last channel local_sum_pix[0][7], to count valid pixels being summed.
			global_sum_rho_sq[rho_global_sum_offset]			= local_sum_rho_sq[lid];
			for (int i=0; i<num_DoFs; i++){
				float4 temp_weights_float4 						= local_sum_weight[i*local_size + lid] / local_size;
				global_sum_weight[se3_global_sum_offset + i] 	= temp_weights_float4 ;

				float4 temp_float4 								= local_sum_grads[i*local_size + lid] / local_size; 	//   / local_sum_grads[i*local_size + lid].w ;  Better to divide by local size, preserve information wrt number of valid pixels.
				global_sum_grads[se3_global_sum_offset + i] 	= temp_float4 ;						// local_sum_grads

				//printf("\nkernel se3_grad_d(..)_2: layer=%i,  group_id=%i,   local_sum_grads[i*local_size + lid]=(%f,%f,%f,%f )", layer, group_id,   temp_float4.x,temp_float4.y,temp_float4.z,temp_float4.w );
			}																						// Save to global_sum_grads // Count hits, and divide group by num hits, without using atomics!
		}else {																						// If no matching pixels in this group, set values to zero.
			global_sum_rho_sq[rho_global_sum_offset]			= 0;
			for (int i=0; i<num_DoFs; i++){
				global_sum_weight[se3_global_sum_offset + i] 	= 0;
				global_sum_grads[se3_global_sum_offset + i] 	= 0;	/*[rho_global_sum_offset]*/
			}
		}
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


__kernel void atomic_test1(
	__private uint num_threads,
	volatile __global int *var_array
		)
{
	uint gid = get_global_id(0);
	if (gid>=num_threads) return;
	int new_var = 1;
	int result = -1;
	result = atomic_add(&var_array[0],  new_var );			// int atomic_add(volatile __global int *p,  int val)
	if (gid == 1) printf("\n__kernel void atomic_test1(..) result = %i",result );
	if (gid==0)return;
	var_array[gid]=new_var;
}

/*
__kernel void atomic_test2(
	__private uint num_threads,
	volatile __global int *var_array
		)
{
	uint gid = get_global_id(0);
	if (gid>=num_threads) return;
	int new_var = 1;
	int result = -1;
//	atomic_fetch_add_explicit( &var_array[0], new_var, memory_order_relaxed ); //atomic_add(&var_array[0],  new_var );			// int atomic_add(volatile __global int *p,  int val)
	if (gid == 1) printf("\n__kernel void atomic_test1(..) result = %i",result );
	if (gid==0)return;
	var_array[gid]=new_var;
}

//atomic_fetch_add_explicit(&acnt, 1, memory_order_relaxed);
*/

