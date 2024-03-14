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
																			// weighting for bi-linear interpolation
	float factor_x = fmod(u_flt,1);
	float factor_y = fmod(v_flt,1);
	c = factor_y * (c_11*factor_x  +  c_01*(1-factor_x))   +   (1-factor_y) * (c_10*factor_x  + c_00*(1-factor_x));
	return c;
}

#endif /*KERNEL_PHOTOMETRIC_COST_CL*/
