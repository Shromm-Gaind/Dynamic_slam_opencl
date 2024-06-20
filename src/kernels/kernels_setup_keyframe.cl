#include "kernels.h"
#include "kernels_macros.h"

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

	//if (global_id_u == 0)printf("\n__kernel void convert_depth(..) invert=%u, factor=%f, depth_mem[global_id_u]=%f,  depth=%f,  1/depth=%f   ", invert, factor, depth_mem[global_id_u], depth, 1/depth  );

	if (!(depth==0)){
		if ( invert==true ) depth_mem_GT[read_index] =  1/depth;
		else depth_mem_GT[read_index] = depth;
	}
}


__kernel void mean_inv_depth(


			  )
{




}


__kernel void normalize_inv_depth(



			  )
{






}
__kernel void transform_cost_volume(
    // inputs
    __private uint mipmap_layer,        //0
    __constant uint8* mipmap_params,    //1
    __constant uint* uint_params,       //2
    __global float* fp32_params,        //3
    __global float16* k2k,              //4
    __global float* old_cdata,          //5 photometric cost volume
    __global float* old_hdata,          //6 hit count volume
    // outputs
    __global float* new_cdata,          //7
    __global float* new_hdata,          //8
    __global float* lo_,                //9 lo, hi, and mean of this ray of the cost volume.
    __global float* hi_,                //10
    __global float* mean_               //11
)
{
    uint global_id_u = get_global_id(0);
    float global_id_flt = global_id_u; // Declare global_id_flt before use

    uint8 mipmap_params_ = mipmap_params[mipmap_layer];
    if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;
    uint read_offset_ = mipmap_params_[MiM_READ_OFFSET];
    uint read_cols_ = mipmap_params_[MiM_READ_COLS];
    uint read_rows_ = mipmap_params_[MiM_READ_ROWS];

    uint mm_cols = uint_params[MM_COLS];
    uint mm_pixels = uint_params[MM_PIXELS]; // Ensure mm_pixels is declared
    uint reduction = mm_cols / read_cols_;
    uint v = global_id_u / read_cols_;   // read_row
    uint u = global_id_u % read_cols_;   // read_column
    float u_flt = u * reduction;         // NB this causes sparse sampling of the original space, to use the same k2k at every scale.
    float v_flt = v * reduction;
    uint read_index = read_offset_ + v * mm_cols + u;

    float16 k2k_pvt = k2k[0];
    float uh2 = k2k_pvt[0] * u_flt + k2k_pvt[1] * v_flt + k2k_pvt[2] * 1;
    float vh2 = k2k_pvt[4] * u_flt + k2k_pvt[5] * v_flt + k2k_pvt[6] * 1;
    float wh2 = k2k_pvt[8] * u_flt + k2k_pvt[9] * v_flt + k2k_pvt[10] * 1;

    if (global_id_u == 0) {
        printf("k2k matrix: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
               k2k_pvt[0], k2k_pvt[1], k2k_pvt[2], k2k_pvt[3], k2k_pvt[4], k2k_pvt[5], k2k_pvt[6], k2k_pvt[7], k2k_pvt[8], k2k_pvt[9], k2k_pvt[10], k2k_pvt[11]);
        printf("uint_params[COSTVOL_LAYERS] = %u\n", uint_params[COSTVOL_LAYERS]);
    }

    float hi, lo, mean, count;
    hi = 0; mean = 0;
    lo = -1;

    // Early boundary check for original coordinates
    bool inbounds = (u_flt >= 0) && (u_flt < read_cols_) && (v_flt >= 0) && (v_flt < read_rows_);
    if (inbounds) {
        if (global_id_u == 10 || global_id_u == 10000 || global_id_u == 20000 || global_id_u == 30000) {
            printf("Valid original coordinates: u_flt = %f, v_flt = %f, global_id_u = %u\n", u_flt, v_flt, global_id_u);
        }

        // Check if cv_layer loop should run
        bool valid_cv_layer_loop = (uint_params[COSTVOL_LAYERS] > 0);

        for (int cv_layer = 0; cv_layer < uint_params[COSTVOL_LAYERS]; cv_layer++) {
            // Check if current layer is valid
            bool valid_layer = (cv_layer < uint_params[COSTVOL_LAYERS]);

            if (valid_layer) {
                float inv_depth = (float)cv_layer * fp32_params[INV_DEPTH_STEP];
                float uh3 = uh2 + k2k_pvt[3] * inv_depth;
                float vh3 = vh2 + k2k_pvt[7] * inv_depth;
                float wh3 = wh2 + k2k_pvt[11] * inv_depth;

                float u3_flt = uh3 / (wh3 * reduction);    // 2D homogeneous pixel coordinates
                float v3_flt = vh3 / (wh3 * reduction);
                float layer_flt = wh3 * inv_depth / fp32_params[INV_DEPTH_STEP]; // 3D depth by layer of old cost volume.

                int write_index = read_index + cv_layer * mm_pixels;

                if (global_id_u == 0 || global_id_u == 10000 || global_id_u == 20000 || global_id_u == 30000) {
                    printf("Global ID: %u, Read Index: %u, Write Index: %u, u3_flt: %f, v3_flt: %f, layer_flt: %f\n",
                           global_id_u, read_index, write_index, u3_flt, v3_flt, layer_flt);
                }

                // If the transformed voxel is within the bounds of the old volume.
                bool inbounds_layer_u3 = (u3_flt >= 0);
                bool inbounds_layer_u4 = (u3_flt < read_cols_);
                bool inbounds_layer_v3 = (v3_flt >= 0);
                bool inbounds_layer_v4 = (v3_flt < read_rows_);
                bool inbounds_layer_lf = (layer_flt >= 0);
                bool inbounds_layer_lf2 = (layer_flt < uint_params[COSTVOL_LAYERS]);
                bool inbounds_layer = inbounds_layer_u3 && inbounds_layer_u4 && inbounds_layer_v3 && inbounds_layer_v4 && inbounds_layer_lf && inbounds_layer_lf2;
                if (inbounds_layer) {
                    // Trilinear interpolation
                    float voxel_c = trilinear(old_cdata, u3_flt, v3_flt, layer_flt, mm_pixels, mm_cols, read_offset_, reduction);
                    float voxel_h = trilinear(old_hdata, u3_flt, v3_flt, layer_flt, mm_pixels, mm_cols, read_offset_, reduction);

                    if (lo == -1) lo = voxel_c;    // TODO should I partially discount previous hits, so they decline in significance?
                    if (voxel_c > hi) hi = voxel_c;
                    if (voxel_c < lo) lo = voxel_c;
                    mean += voxel_c;  // new_cdata[write_index];
                    count++;
                    new_cdata[write_index] += voxel_c;
                    new_hdata[write_index] += voxel_h;

                    if (global_id_u == 0 || global_id_u == 10000 || global_id_u == 20000 || global_id_u == 30000) {
                        printf("Valid read/write: u3_flt = %f, v3_flt = %f, layer_flt = %f, global_id_u = %u, voxel_c = %f, voxel_h = %f\n",
                               u3_flt, v3_flt, layer_flt, global_id_u, voxel_c, voxel_h);
                    }
                } else {
                    if (global_id_u == 0 || global_id_u == 10000 || global_id_u == 20000 || global_id_u == 30000) {
                        printf("Invalid read/write: u3_flt = %f, v3_flt = %f, layer_flt = %f, global_id_u = %u\n", u3_flt, v3_flt, layer_flt, global_id_u);
                    }
                }
            }
        }
        // Set hi_mem, lo_mem for new cost_vol
        mean /= count;
        lo_[read_index] = lo;
        hi_[read_index] = hi;
        mean_[read_index] = mean;
    } else {
        if (global_id_u == 0 || global_id_u == 10000 || global_id_u == 20000 || global_id_u == 30000) {
            printf("Invalid original coordinates: u_flt = %f, v_flt = %f, global_id_u = %u\n", u_flt, v_flt, global_id_u);
        }
    }
}



__kernel void transform_depthmap(  // Needs to transform as a point cloud. Need to compare & swap the incomming points on the new depthmap.
	// inputs
	__private	uint	mipmap_layer,			//0
	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__global 	float16*k2k,					//3
	__global 	float4*	old_keyframe,			//4
	__global	float* 	depth_map_in,			//5
	// output
	__global	float* 	depth_map_out			//6
		)
{
	uint global_id_u 	= get_global_id(0);

    if (global_id_u == 0) {
    printf("Kernel transform_cost_volume started\n");
    }

	float global_id_flt = global_id_u;
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

	//if (global_id_u ==0) printf("\n__kernel void transform_depthmap(): k2k_pvt= %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f, ",\
	//	k2k_pvt[0], k2k_pvt[1], k2k_pvt[2], k2k_pvt[3], k2k_pvt[4], k2k_pvt[5], k2k_pvt[6], k2k_pvt[7], k2k_pvt[8], k2k_pvt[9], k2k_pvt[10], k2k_pvt[11]
	//);

	float uh2 = k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1 + k2k_pvt[3]*inv_depth;
	float vh2 = k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1 + k2k_pvt[7]*inv_depth;
	float wh2 = k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1+ k2k_pvt[11]*inv_depth;
	//float h/z  = k2k_pvt[12]*u_flt + k2k_pvt[13]*v + k2k_pvt[14]*1; // +k2k_pvt[15]/z

	float u2_flt		= uh2/wh2;
	float v2_flt		= vh2/wh2;
	float newdepth		= wh2*inv_depth;															// 3D depth transformed to new keyframe.

	int   u2			= floor((u2_flt/reduction)+0.5f) ;											// nearest neighbour interpolation
	int   v2			= floor((v2_flt/reduction)+0.5f) ;											// NB this corrects the sparse sampling to the redued scales.
	uint write_index 	= read_offset_ + v2 * mm_cols  + u2; // read_cols_
	if (global_id_u >= layer_pixels  || u2<0 || u2>read_cols_ || v2<0 || v2>read_rows_ ) return;
	////////////////////////////////////////////////////////
																									// TODO should I use more sophisticated interpolation ?
	if(alpha > 0){																					// alpha indicates if this pixel of new depth map has valid source in the old depth map.
		atomic_maxf(&depth_map_out[write_index],  newdepth );//map holds inv_depth, hence atomic_maxf(..)	// NB depth_map may have occlusions when transformed.
																									// Alternatives would be
																									// (i) to raytrace theough the costvol
																									// (ii) Import the Amem debug costvol.
	}
	//depth_map_out[read_index] = newdepth;// vh2/(read_rows_*256);
	// uh2/(read_cols_*256); //v2_flt;// u2_flt;// vh2/read_cols_; // uh2/read_cols_; // ((float)read_index_new)/((float)mm_pixels); // newdepth; //depth_map_in[read_index]; //
}


