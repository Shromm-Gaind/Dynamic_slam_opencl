#ifndef KERNEL_PHOTOMETRIC_COST_CL

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
	int int_u2 = ceil(u_flt/reduction-0.5f);									// nearest neighbour interpolation
	int int_v2 = ceil(v_flt/reduction-0.5f);									// NB this corrects the sparse sampling to the redued scales.
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


float8 bilinear_SE3_grad (__global float8* img, float u_flt, float v_flt, int cols, int read_offset_){ 								//, uint reduction, int i, int mm_pixels){
	float8 	c, c_00, c_01, c_10, c_11;
	int coff_00, coff_01, coff_10, coff_11;
	int int_u2 = ceil(u_flt);
	int int_v2 = ceil(v_flt);
																			// compute adjacent pixel indices & sample adjacent pixels
	c_11 = img[ read_offset_ + int_v2     * cols +  int_u2     ];
	c_10 = img[ read_offset_ + (int_v2-1) * cols +  int_u2     ];
	c_01 = img[ read_offset_ + int_v2     * cols + (int_u2 -1) ];
	c_00 = img[ read_offset_ + (int_v2-1) * cols + (int_u2 -1) ];

	uint  global_id_u 	= get_global_id(0);
	//if(global_id_u == 10000  ){ printf("\n__bilinear_SE3_grad (global_id_u == 10000 )  chk_1  , coff_00=%u",  read_offset_ + (int_v2-1) * cols + (int_u2 -1)   ); }

																			// weighting for bi-linear interpolation
	float factor_x = fmod(u_flt,1);
	float factor_y = fmod(v_flt,1);
	c = factor_y * (c_11*factor_x  +  c_01*(1-factor_x))   +   (1-factor_y) * (c_10*factor_x  + c_00*(1-factor_x));
	return c;
}

float4 bilinear_flt4 (__global float4* img, float u_flt, float v_flt, int cols, int read_offset_){                                   // Used in tracking
	float4 	c, c_00, c_01, c_10, c_11;										// read_offset_ + v2 * mm_cols  + u2;
	int coff_00, coff_01, coff_10, coff_11;
	int int_u2 = ceil(u_flt);
	int int_v2 = ceil(v_flt);
																			// compute adjacent pixel indices & sample adjacent pixels
	c_11 = img[ read_offset_ + int_v2     * cols +  int_u2     ];
	c_10 = img[ read_offset_ + (int_v2-1) * cols +  int_u2     ];
	c_01 = img[ read_offset_ + int_v2     * cols + (int_u2 -1) ];
	c_00 = img[ read_offset_ + (int_v2-1) * cols + (int_u2 -1) ];

	uint  global_id_u 	= get_global_id(0);
	//if(global_id_u == 10000  ){ printf("\n__bilinear_flt4 (global_id_u == 10000 )  chk_1  , coff_00=%u",  read_offset_ + (int_v2-1) * cols + (int_u2 -1)   ); }

																			// weighting for bi-linear interpolation
	float factor_x = fmod(u_flt,1);
	float factor_y = fmod(v_flt,1);
	c = factor_y * (c_11*factor_x  +  c_01*(1-factor_x))   +   (1-factor_y) * (c_10*factor_x  + c_00*(1-factor_x));
	return c;
}

void bilinear_SE3_grad_weight (float4 weights[6],
							   __global float8* 	SE3_grad_map_cur_frame,
							   int 					read_index,
							   __global float8* 	SE3_grad_map_new_frame,
							   float 				u2_flt,
							   float 				v2_flt,
							   int 					cols,
							   int 					read_offset_,
							   uint 				reduction,
							   uint 				mm_pixels,
							   float 				alpha
							   /*, int channel*/ ){


// TODO  problem some results are NaN
	// NB for each se3_dim:  weight = 1/ (SE3_grad_map_cur_frame - SE3_grad_map_new_frame)
	float8 	c, c_00, c_01, c_10, c_11;

	int int_u2 					= ceil(u2_flt);
	int int_v2 					= ceil(v2_flt);

	int coff_00					= read_offset_ + (int_v2-1) * cols + (int_u2 -1) ;
	int coff_01					= read_offset_ + int_v2     * cols + (int_u2 -1) ;
	int coff_10					= read_offset_ + (int_v2-1) * cols +  int_u2 ;
	int coff_11					= read_offset_ + int_v2     * cols +  int_u2 ;

	uint  global_id_u 	= get_global_id(0);
	//if(global_id_u == 1000  ){ printf("\n__bilinear_SE3_grad_weight (global_id_u == 1000 )  chk_1  , coff_00=%u",  coff_00   ); }


	float factor_x 				= fmod(u2_flt,1);
	float factor_y 				= fmod(v2_flt,1);
	float n_factor_x			= 1-factor_x;
	float n_factor_y			= 1-factor_y;

	for (int se3_dim = 0; se3_dim<6; se3_dim++){

		int dim_offset 			= se3_dim * mm_pixels;
		float8 SE3_grad_cur_v8 	= SE3_grad_map_cur_frame[read_index + dim_offset];
		float SE3_grad_cur_[8];
		vstore8(SE3_grad_cur_v8,0,SE3_grad_cur_);

		c_11 					= SE3_grad_map_new_frame[ dim_offset + coff_11 ];
		c_10 					= SE3_grad_map_new_frame[ dim_offset + coff_10 ];
		c_01 					= SE3_grad_map_new_frame[ dim_offset + coff_01 ];
		c_00 					= SE3_grad_map_new_frame[ dim_offset + coff_00 ];

		float8 SE3_grad_new_v8 	= factor_y * (c_11*factor_x  +  c_01*n_factor_x)   +   n_factor_y * (c_10*factor_x  + c_00*n_factor_x);
		float SE3_grad_new_[8];
		vstore8(SE3_grad_new_v8, 0, SE3_grad_new_);			// vstore4(int_vec, 0, int_array);

		float weights_[4];
		for (int chan = 0; chan < 3; chan++){	//int chan = channel;
			weights_[chan]									= SE3_grad_cur_[chan] + SE3_grad_cur_[chan + 4] -  SE3_grad_new_[chan] - SE3_grad_new_[chan + 4];
			weights_[chan] 									= 1/weights_[chan];
			if (!isnormal(weights_[chan])) 	weights_[chan] 	= 0;
		}
		weights_[3]		= alpha;

		float4 weights4 = { weights_[0], weights_[1], weights_[2], weights_[3]   };
		weights[se3_dim] = weights4;
	}
}

/*
float bilinear_grad_weight (__global float8* SE3_grad_map_cur_frame, __global float8* SE3_grad_map_new_frame, int read_index, float u2_flt, float v2_flt, int cols, int read_offset_, uint reduction){

	float8 HSV_grad_ 		= HSV_grad[read_index];
	float value_grad_u		= HSV_grad_[6];
	float value_grad_v		= HSV_grad_[7];

	float8 HSV_grad2_ 		= bilinear (HSV_grad, u2_flt, v2_flt, cols, read_offset_, reduction);
	float value_grad2_u		= HSV_grad2_[6];
	float value_grad2_v		= HSV_grad2_[7];

	float weight 			= sqrt(  pown((value_grad_u - value_grad2_u), 2) + pown((value_grad_v - value_grad2_v), 2) );
	if (weight > 2/FLT_MAX) return 1/weight;
	else return 0;
}
*/

/*
float4 compute_se3_incr (float rho, float weight,    ){


		float8 HSV_grad_ 					= HSV_grad[read_index];
		float8 SE3_grad_map_cur_frame_		= SE3_grad_map_cur_frame[read_index];
		se3_incr 							= weight * rho / (SE3_grad_map_cur_frame_ * value_grad);	//

}
*/
#endif /*KERNEL_PHOTOMETRIC_COST_CL*/
