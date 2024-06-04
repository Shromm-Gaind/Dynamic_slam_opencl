#include "RunCL.hpp"

void RunCL::computeSigmas(float epsilon, float theta, float L, float &sigma_d, float &sigma_q ){
	int local_verbosity_threshold = verbosity_mp["RunCL::computeSigmas"];

		float mu	= 2.0*std::sqrt((1.0/theta)*epsilon) /L;
		sigma_d		=  mu / (2.0/ theta)  ;
		sigma_q 	=  mu / (2.0*epsilon) ;
}

void RunCL::updateDepthCostVol(cv::Matx44f K2K_, int count, uint start, uint stop){ //buildDepthCostVol();
	int local_verbosity_threshold = verbosity_mp["RunCL::updateDepthCostVol"];// -1;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateDepthCostVol(..)_chk0 ."<<flush;}
	save_index = keyFrameCount*1000 + costvol_frame_num;

	cl_event writeEvt;
	cl_int status;
	float K2K_arry[16]; for (int i=0; i<16;i++){ K2K_arry[i] = K2K_.operator()(i/4,i%4); }

	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * sizeof(float), K2K_arry, 		0, NULL, &writeEvt);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush;
																																										cout << "Error: RunCL::updateDepthCostVol(..)_chk0.5\t" << "save_index_" <<save_index << endl;exit_(status);}
	clFlush(uload_queue); status = clFinish(uload_queue);
	//waitForEventAndRelease( &writeEvt );
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateDepthCostVol(..)_chk0.7 \t" << "save_index_" <<save_index<<flush;}
	cl_int 				res;
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                        //__private	    uint	    layer,			//0
	res = clSetKernelArg(depth_cost_vol_kernel,  1, sizeof(cl_mem), &mipmap_buf);			if(res!=CL_SUCCESS){cout<<"\nmipmap_buf = "			<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	//1
	res = clSetKernelArg(depth_cost_vol_kernel,  2, sizeof(cl_mem), &uint_param_buf);		if(res!=CL_SUCCESS){cout<<"\nuint_param_buf = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,	//2
	res = clSetKernelArg(depth_cost_vol_kernel,  3, sizeof(cl_mem), &fp32_param_buf);		if(res!=CL_SUCCESS){cout<<"\nfp32_param_buf = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params,	//3
	res = clSetKernelArg(depth_cost_vol_kernel,  4, sizeof(cl_mem), &k2kbuf);				if(res!=CL_SUCCESS){cout<<"\nk2kbuf = "				<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float16*	k2k,			//4
	res = clSetKernelArg(depth_cost_vol_kernel,  5, sizeof(cl_mem), &keyframe_imgmem_HSV_grad);		if(res!=CL_SUCCESS){cout<<"\nkeyframe_basemem = "	<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float8* 	base,			//5		keyframe_basemem

	res = clSetKernelArg(depth_cost_vol_kernel,  6, sizeof(cl_mem), &HSV_grad_mem/*imgmem*/);if(res!=CL_SUCCESS){cout<<"\nHSV_grad_mem = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float8* 	img,			//6		HSV_grad_mem/*imgmem*/ now float8
	res = clSetKernelArg(depth_cost_vol_kernel,  7, sizeof(cl_mem), &cdatabuf);				if(res!=CL_SUCCESS){cout<<"\ncdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  	cdata,			//7
	res = clSetKernelArg(depth_cost_vol_kernel,  8, sizeof(cl_mem), &hdatabuf);				if(res!=CL_SUCCESS){cout<<"\nhdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  	hdata,			//8
	res = clSetKernelArg(depth_cost_vol_kernel,  9, sizeof(cl_mem), &lomem);				if(res!=CL_SUCCESS){cout<<"\nlomem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  	lo,				//9
	res = clSetKernelArg(depth_cost_vol_kernel, 10, sizeof(cl_mem), &himem);				if(res!=CL_SUCCESS){cout<<"\nhimem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  	hi,				//10
	res = clSetKernelArg(depth_cost_vol_kernel, 11, sizeof(cl_mem), &amem);					if(res!=CL_SUCCESS){cout<<"\namem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  	a,				//11	amem, auxilliary A
	res = clSetKernelArg(depth_cost_vol_kernel, 12, sizeof(cl_mem), &dmem);					if(res!=CL_SUCCESS){cout<<"\ndmem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  	d,				//12	dmem, depth D
	res = clSetKernelArg(depth_cost_vol_kernel, 13, sizeof(cl_mem), &img_sum_buf);			if(res!=CL_SUCCESS){cout<<"\nimg_sum_buf res = " 	<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  	img_sum,		//13
	res = clSetKernelArg(depth_cost_vol_kernel, 14, sizeof(cl_mem), &cdatabuf_8chan);		if(res!=CL_SUCCESS){cout<<"\ncdatabuf_8chan res = " <<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float8* 	cdata_8chan		//14

	// res = clSetKernelArg(img_grad_kernel, 10, sizeof(cl_mem), &HSV_grad_mem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}						//__global 	 float4*	HSV_grad_mem//10
	/*
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

	#define IMG_MEAN			0	// for img_stats
	#define IMG_VAR 			1	//
	*/

	/* param_buf , from from DTAM_opencl : RunCL::calcCostVol(float* k2k,  cv::Mat &image) -> __kernel void BuildCostVolume2(..)
	#define pixels_			0  // Can these be #included from a common header for both host and device code?
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
	*/

	/* from DTAM_opencl : RunCL::calcCostVol(float* k2k,  cv::Mat &image)
	res = clSetKernelArg(cost_kernel, 0, sizeof(cl_mem), &k2kbuf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(cost_kernel, 2, sizeof(cl_mem), &imgmem);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	*/

	/* from DTAM_opencl : RunCL::allocatemem(..)
	res = clSetKernelArg(cost_kernel, 1, sizeof(cl_mem),  &basemem);		if(res!=CL_SUCCESS){cout<<"\nbasemem res= "   		<<checkerror(res)<<"\n"<<flush;exit_(res);} // base
	res = clSetKernelArg(cost_kernel, 3, sizeof(cl_mem),  &cdatabuf);		if(res!=CL_SUCCESS){cout<<"\ncdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);} // cdata
	res = clSetKernelArg(cost_kernel, 4, sizeof(cl_mem),  &hdatabuf);		if(res!=CL_SUCCESS){cout<<"\nhdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);} // hdata
	res = clSetKernelArg(cost_kernel, 5, sizeof(cl_mem),  &lomem);			if(res!=CL_SUCCESS){cout<<"\nlomem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);} // lo
	res = clSetKernelArg(cost_kernel, 6, sizeof(cl_mem),  &himem);			if(res!=CL_SUCCESS){cout<<"\nhimem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);} // hi
	res = clSetKernelArg(cost_kernel, 7, sizeof(cl_mem),  &amem);			if(res!=CL_SUCCESS){cout<<"\namem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);} // a
	res = clSetKernelArg(cost_kernel, 8, sizeof(cl_mem),  &dmem);			if(res!=CL_SUCCESS){cout<<"\ndmem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);} // d
	res = clSetKernelArg(cost_kernel, 9, sizeof(cl_mem),  &param_buf);		if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "		<<checkerror(res)<<"\n"<<flush;exit_(res);} // param_buf
	res = clSetKernelArg(cost_kernel,10, sizeof(cl_mem),  &img_sum_buf);	if(res!=CL_SUCCESS){cout<<"\nimg_sum_buf res = " 	<<checkerror(res)<<"\n"<<flush;exit_(res);} // cdata
																				if(verbosity>0) cout << "RunCL::allocatemem_finished\n\n" << flush;
	*/

	// ? use depth_mem[!frame_bool_idx]/[frame_bool_idx]  ?  Also for other params ?
	/*
	__kernel void BuildCostVolume2(						// called as "cost_kernel" in RunCL.cpp
	// TODO rewrite with homogeneuos coords to handle points at infinity (x,y,z,0) -> (u,v,0)
	__global float* k2k,		//0
	__global float* base,		//1  // uchar*
	__global float* img,		//2  // uchar*
	__global float* cdata,		//3
	__global float* hdata,		//4  'w' num times cost vol elem has been updated
	__global float* lo, 		//5
	__global float* hi,			//6
	__global float* a,			//7
	__global float* d,			//8
	__constant float* params,	//9  pixels, rows, cols, layers, max_inv_depth, min_inv_depth, inv_d_step, threshold
	__global float* img_sum)	//10
{
	*/
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateDepthCostVol(..)_chk1 ."<<flush;}
	mipmap_call_kernel( depth_cost_vol_kernel, m_queue, start, stop ); // , true
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateDepthCostVol(..)_chk3 ."<<flush;}
																																			if(verbosity>local_verbosity_threshold ) { // && count == costVolLayers - 1
																																				cout << "\ncount = " << count << flush;
																																				stringstream ss;
																																				ss << "buildDepthCostVol" << save_index;													// Save buffers to file ###########
																																				bool show = false;
																																				bool exception_tiff = false;
																																				//DownloadAndSave_3Channel(	imgmem,  			ss.str(), paths.at("imgmem"), 			mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	false );
																																				DownloadAndSave_HSV_grad(  HSV_grad_mem/*imgmem*/, 	ss.str(), paths.at("HSV_grad_mem"),	mm_size_bytes_C8,   mm_Image_size,   CV_32FC(8),show, -1, 0 );
																																				DownloadAndSave(			lomem,  			ss.str(), paths.at("lomem"),  			mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , 8); // a little more than the num images in costvol.
																																				DownloadAndSave(		 	himem,  			ss.str(), paths.at("himem"),  			mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , 8); //params[COSTVOL_LAYERS]
																																				DownloadAndSave(		 	amem,   			ss.str(), paths.at("amem"),   			mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , fp32_params[MAX_INV_DEPTH]);
																																				DownloadAndSave(		 	dmem,   			ss.str(), paths.at("dmem"),   			mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , fp32_params[MAX_INV_DEPTH]);
																																				if (verbosity_mp["RunCL::updateDepthCostVol::cdatabuf"])
																																					DownloadAndSaveVolume(		cdatabuf, 			ss.str(), paths.at("cdatabuf"), 		mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 /*TODO count*/ , exception_tiff /*exception_tiff=false*/);
																																				if (verbosity_mp["RunCL::updateDepthCostVol::cdatabuf_8chan"])
																																					DownloadAndSave_8Channel_volume( cdatabuf_8chan,ss.str(), paths.at("cdatabuf_8chan"),	mm_size_bytes_C8,	mm_Image_size,   CV_32FC1,  show , 1, costVolLayers );
																																				if (verbosity_mp["RunCL::updateDepthCostVol::hdatabuf"])
																																					DownloadAndSaveVolume(		hdatabuf, 			ss.str(), paths.at("hdatabuf"), 		mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 /*TODO count*/ , exception_tiff /*exception_tiff=false*/);
																																				if(verbosity>1) cout << "\ncostvol_frame_num="<<costvol_frame_num;
																																				cout << "\nRunCL::updateDepthCostVol(..)_finished\n" << flush;
																																			}
}

void RunCL::updateQD(float epsilon, float theta, float sigma_q, float sigma_d, uint start, uint stop){
	int local_verbosity_threshold = verbosity_mp["RunCL::updateQD"];// -1;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateQD(..)_chk0 ."<<flush;}
	QD_count++;

	fp32_params[EPSILON]		=  epsilon;
	fp32_params[SIGMA_Q]		=  sigma_q;
	fp32_params[SIGMA_D]		=  sigma_d;
	fp32_params[THETA]			=  theta;

	cl_int status, res;
	cl_event writeEvt, ev;
	status = clEnqueueWriteBuffer(uload_queue, fp32_param_buf, CL_FALSE, 0, 16 * sizeof(float), fp32_params, 0, NULL, &writeEvt); 			// WriteBuffer param_buf ##########
																				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: updateQD_chk0\n"	<< endl; exit_(status);}
	status = clFlush(uload_queue); 												if (status != CL_SUCCESS)	{ cout << "\nclFlush status = " 				<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = waitForEventAndRelease(&writeEvt); 								if (status != CL_SUCCESS)	{ cout << "\nwaitForEventAndRelease status = " 	<< checkerror(status) <<"\n"<<flush; exit_(status);}

	// __private	uint	layer	, set in mipmap_call_kernel(..) below																									//__private	uint	layer,				//0
	res = clSetKernelArg(updateQD_kernel, 1, sizeof(cl_mem), &mipmap_buf);		if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__constant 	uint*	mipmap_params,	//1
	res = clSetKernelArg(updateQD_kernel, 2, sizeof(cl_mem), &uint_param_buf);	if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__constant 	uint*	uint_params,	//2
	res = clSetKernelArg(updateQD_kernel, 3, sizeof(cl_mem), &fp32_param_buf);	if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global 	float*  fp32_params,		//3
	res = clSetKernelArg(updateQD_kernel, 4, sizeof(cl_mem), &keyframe_g1mem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res) <<"\n"<<flush;exit_(res);}			//__global 	float4* g1pt,				//4
	res = clSetKernelArg(updateQD_kernel, 5, sizeof(cl_mem), &qmem);	 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res) <<"\n"<<flush;exit_(res);}			//__global 	float* 	qpt,				//5		// qmem,						//	2 * mm_size_bytes_C1
	res = clSetKernelArg(updateQD_kernel, 6, sizeof(cl_mem), &amem);	 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res) <<"\n"<<flush;exit_(res);}			//__global 	float*  apt,				//6		// amem,     auxilliary A		//	mm_size_bytes_C1
	res = clSetKernelArg(updateQD_kernel, 7, sizeof(cl_mem), &dmem);	 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res) <<"\n"<<flush;exit_(res);}			//__global 	float*  dpt					//7		// dmem,     depth D			//	mm_size_bytes_C1

	//res = clSetKernelArg(updateQD_kernel, 8, sizeof(cl_mem), &qmem2);	 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res) <<"\n"<<flush;exit_(res);}			//__global 	float* 	qpt,				//8		// qmem2,						//	2 * mm_size_bytes_C1


																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateQD(..)_chk1 ."<<flush;}
	mipmap_call_kernel( updateQD_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::updateQD(..)_chk3, epsilon="<<epsilon<<"  , sigma_Q="<<sigma_q<<"  , sigma_D="<<sigma_d<<"  , theta="<<theta<<" ."<<flush;
																																			}
																																			if(verbosity>local_verbosity_threshold){
																																				stringstream ss;
																																				ss << "updateQD"<< save_index << "_QD_count_" << QD_count <<"_epsilon_"<<epsilon<<"_sigmaQ_"<<sigma_q<<"_D_"<<sigma_d<<"_theta_"<<theta;
																																				//int this_count = save_index * 1000 + QD_count;
																																				//ss << save_index << "_QD_count_" << QD_count;

																																				cv::Size q_size( mm_Image_size.width, 2* mm_Image_size.height ); 			// 2x sized for qx and qy.
																																				DownloadAndSave(dmem,   ss.str(), paths.at("dmem"),    mm_size_bytes_C1 , mm_Image_size , CV_32FC1, false ,    fp32_params[MAX_INV_DEPTH]  );
																																				DownloadAndSave(qmem,   ss.str(), paths.at("qmem"),  2*mm_size_bytes_C1 , q_size        , CV_32FC1, false , -1*fp32_params[MAX_INV_DEPTH]  );

																																				//DownloadAndSave(qmem2,   ss.str(), paths.at("qmem2"),2*mm_size_bytes_C1 , q_size        , CV_32FC1, false , -1*fp32_params[MAX_INV_DEPTH]  );  // 1/uint_params[MM_PIXELS]

																																				cout<<"\nRunCL::updateQD_chk3_finished\n"<<flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateQD(..)_finished ."<<flush;}
}

void RunCL::updateG(int count, uint start, uint stop){
	int local_verbosity_threshold = verbosity_mp["RunCL::updateG"];// -1;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateG(..)_chk0"<<flush;}
	G_count++;
	cl_int res;
	size_t num_threads = ceil( (float)(mm_layerstep)/(float)local_work_size ) * local_work_size ;
																																			if(verbosity>local_verbosity_threshold) { cout<<"\n\nRunCL::updateG(..)_chk1"<<flush;
																																				cout << ",   num_threads = " << num_threads << ",   mm_layerstep = " << mm_layerstep << ",  local_work_size = " << local_work_size  <<endl << flush;}
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                            //__private	 uint	    layer,			//0
    res = clSetKernelArg(updateG_kernel, 1, sizeof(cl_mem), &mipmap_buf);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*		mipmap_params,	//1
	res = clSetKernelArg(updateG_kernel, 2, sizeof(cl_mem), &uint_param_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*		uint_params		//2
	res = clSetKernelArg(updateG_kernel, 3, sizeof(cl_mem), &fp32_param_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant float*		fp32_params		//3
	res = clSetKernelArg(updateG_kernel, 4, sizeof(cl_mem), &keyframe_imgmem_HSV_grad);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global   float8*	img,			//4
	res = clSetKernelArg(updateG_kernel, 5, sizeof(cl_mem), &keyframe_g1mem);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float8*	g1p				//5

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateG(..)_chk2"<<flush;}
	mipmap_call_kernel( updateG_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateG(..)_chk3 Saving keyframe_g1mem."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_updateG";
																																				/*
																																					stringstream ss_path;
																																					ss_path << "keyframe_g1mem";
																																					cout << "\n" << ss_path.str() 			<<flush;
																																					cout << "\n" << paths.at(ss_path.str()) <<flush;
																																				*/
																																				float max_range = 1.0;   // produce tiff with exact float values from the buffer.
																																				DownloadAndSave_HSV_grad(  keyframe_g1mem,	ss.str(), paths.at("keyframe_g1mem"), 	mm_size_bytes_C8, mm_Image_size,  CV_32FC(8),false, max_range, 0 );	//  /* paths.at(ss_path.str()) */// CV_32FC4, 	false );
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateG(..)_chk4 Finished."<<flush;}
}

void RunCL::updateA(float lambda, float theta,  uint start, uint stop){
	int local_verbosity_threshold = verbosity_mp["RunCL::updateA"];// -1;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateA(..)_chk0 ."<<flush;}
	A_count++;

	fp32_params[THETA]			=  theta;
	fp32_params[LAMBDA]			=  lambda;
	cl_int status;
	cl_event writeEvt;
	status = clEnqueueWriteBuffer(uload_queue,  fp32_param_buf, CL_FALSE, 0, 16 * sizeof(float), fp32_params, 0, NULL, &writeEvt);										// WriteBuffer param_buf ##########
												if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: \nRunCL::updateA_chk0\n" << endl; exit_(status);}
																																			else if(verbosity>0) {cout << "\nRunCL::updateA_chk0.5\t\tlayers="<< fp32_params[COSTVOL_LAYERS] <<" \n" << flush;}

	status = clFlush(uload_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush status = " << status << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = waitForEventAndRelease(&writeEvt); if (status != CL_SUCCESS)	{ cout << "\nwaitForEventAndRelease status = "<<status<<checkerror(status)<<"\n"<<flush; exit_(status);}


	cl_int		res;
	cl_event	ev;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateA(..)_chk1 ."<<flush;}

	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                            //__private	 uint	    layer,			//0
	res = clSetKernelArg(updateA_kernel, 1, sizeof(cl_mem), &mipmap_buf); 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant 	uint*	mipmap_params,	//1
	res = clSetKernelArg(updateA_kernel, 2, sizeof(cl_mem), &uint_param_buf); 	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant 	uint*	uint_params,	//2
	res = clSetKernelArg(updateA_kernel, 3, sizeof(cl_mem), &fp32_param_buf); 	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  fp32_params,		//3
	res = clSetKernelArg(updateA_kernel, 4, sizeof(cl_mem), &cdatabuf); 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  cdata,				//4		//           cost volume
	res = clSetKernelArg(updateA_kernel, 5, sizeof(cl_mem), &lomem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  lo,					//5
	res = clSetKernelArg(updateA_kernel, 6, sizeof(cl_mem), &himem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  hi,					//6
	res = clSetKernelArg(updateA_kernel, 7, sizeof(cl_mem), &amem);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  apt,				//7		// amem,     auxilliary A
	res = clSetKernelArg(updateA_kernel, 8, sizeof(cl_mem), &dmem);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  dpt					//8		// dmem,     depth D

	res = clSetKernelArg(updateA_kernel, 9, sizeof(cl_mem), &dbg_databuf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  dbg_data			//9		// dbg_databuf,   debugging buffer

	status = clFlush(m_queue); 					if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateA(..)_chk2 ."<<flush;}
	mipmap_call_kernel( updateA_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::updateA(..)_chk3, A_count=" << A_count << ", theta=" << theta << ", lambda="<< lambda<<flush;
																																			}
																																			if(A_count%1==0 && verbosity>local_verbosity_threshold){
																																				stringstream ss;
																																				ss << "updateA"<< save_index << "_A_count_" << A_count << "_theta_" << theta << "_lambda_"<< lambda ;
																																				//count = keyFrameCount*1000000 + A_count*1000 + 999;
																																				//ss << count << "_theta"<<theta<<"_";

																																				cv::Size q_size( mm_Image_size.width, 2* mm_Image_size.height );
																																				DownloadAndSave(amem,   ss.str(), paths.at("amem"),    mm_size_bytes_C1,   mm_Image_size, CV_32FC1,  false , fp32_params[MAX_INV_DEPTH]);
																																				DownloadAndSave(dmem,   ss.str(), paths.at("dmem"),    mm_size_bytes_C1,   mm_Image_size, CV_32FC1,  false , fp32_params[MAX_INV_DEPTH]);
																																				DownloadAndSave(qmem,   ss.str(), paths.at("qmem"),  2*mm_size_bytes_C1,   q_size       , CV_32FC1,  false , -1*fp32_params[MAX_INV_DEPTH] ); //0.1

																																				if (verbosity_mp["RunCL::updateA::dbg_databuf"]){
																																					bool show = false;
																																					bool exception_tiff = false;
																																					DownloadAndSaveVolume(dbg_databuf, 	ss.str(), paths.at("dbg_databuf"), 	mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 /*TODO count*/ , exception_tiff );
																																				}
																																			}
																																			if(verbosity>0) cout<<"\nRunCL::updateA_chk2_finished,   fp32_params[MAX_INV_DEPTH]="<<fp32_params[MAX_INV_DEPTH]<<flush;
}

void RunCL::measureDepthFit(uint start, uint stop){
	int local_verbosity_threshold = verbosity_mp["RunCL::measureDepthFit"];// -1;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::measureDepthFit(..)_chk0 ."<<flush;}

	cl_int 		status;
	cl_int		res;
	cl_event	ev;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::measureDepthFit(..)_chk1 ."<<flush;}

	res = clSetKernelArg(measureDepthFit_kernel, 1, sizeof(cl_mem), &mipmap_buf); 			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant 	uint*	mipmap_params,				//1
	res = clSetKernelArg(measureDepthFit_kernel, 2, sizeof(cl_mem), &uint_param_buf); 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant 	uint*	uint_params,				//2
	res = clSetKernelArg(measureDepthFit_kernel, 3, sizeof(cl_mem), &fp32_param_buf); 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  fp32_params,				//3

	res = clSetKernelArg(measureDepthFit_kernel, 4, sizeof(cl_mem), &dmem);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  dpt							//4		// dmem,     depth D
	res = clSetKernelArg(measureDepthFit_kernel, 5, sizeof(cl_mem), &keyframe_depth_mem_GT);if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  dpt_GT						//5
	res = clSetKernelArg(measureDepthFit_kernel, 6, sizeof(cl_mem), &dmem_disparity);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*  dpt_disparity				//6

	res = clSetKernelArg(measureDepthFit_kernel, 7, local_work_size*4*sizeof(float), NULL);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float*	local_sum_dpt_disparity		//7
	res = clSetKernelArg(measureDepthFit_kernel, 8, sizeof(cl_mem), &dmem_disparity_sum);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float*	global_sum_dpt_disparity,	//8


	status = clFlush(m_queue); 					if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::measureDepthFit(..)_chk2 ."<<flush;}
	mipmap_call_kernel( measureDepthFit_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::measureDepthFit(..)_chk3, A_count=" << A_count <<flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {
																																				stringstream ss;
																																				ss << "measureDepthFit"<< save_index << "_A_count_" << A_count ;
																																				DownloadAndSave_3Channel(dmem_disparity,   ss.str(), paths.at("dmem_disparity"),    mm_size_bytes_C4,   mm_Image_size, CV_32FC4,  false , 1.0,   0 ,  false); //  float max_range /*=1*/, uint offset /*=0*/, bool exception_tiff /*=false*/)
																																			}
	cv::Mat dmem_disparity_sum_mat = cv::Mat::zeros (d_disp_sum_size, 1, CV_32FC4); // cv::Mat::zeros (int rows, int cols, int type)		// NB the data returned is one float4 per group, for the base image, holding disparity (depth, ....) plus entry[3]=pixel count.
	ReadOutput( dmem_disparity_sum_mat.data, var_sum_mem, d_disp_sum_size_bytes );                                                          // se3_sum_size_bytes
	                                                                                                                                         if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::measureDepthFit(..)_chk3 ."<<flush;
																																				cout << "\ndmem_disparity_sum_mat.size()="<<dmem_disparity_sum_mat.size()<<flush;
																																				cout << "\nd_disp_sum_size="<<d_disp_sum_size<<flush;
                                                                                                                                                cout << "\n dmem_disparity_sum_mat.data = (\n";

                                                                                                                                                for (int i=0; i<d_disp_sum_size; i++){
                                                                                                                                                    cout << "\n group="<<i<<" : ( " << flush;
                                                                                                                                                    for (int j=0; j<4; j++){
                                                                                                                                                        cout << dmem_disparity_sum_mat.at<float>(i,j) << " , " << flush;
                                                                                                                                                    }
                                                                                                                                                    cout << ")" << flush;
                                                                                                                                                }cout << "\n)\n" << flush;
                                                                                                                                            }
	// TODO   get summation of depth error working
	float depth_disp_sum_reults[4] = {0};
	uint groups_to_sum = dmem_disparity_sum_mat.at<float>(0, 0);
	uint start_group   = 1;
	uint stop_group    = start_group + groups_to_sum;
																																			if(verbosity>local_verbosity_threshold) cout << "\ngroups_to_sum="<<groups_to_sum<<",  stop_group="<<stop_group<<endl<<flush;
	for (int j=start_group; j< stop_group; j++){
		for (int k=0; k<4; k++){
			depth_disp_sum_reults[k] += dmem_disparity_sum_mat.at<float>(j, k);
		}
	}
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n start_group=" << start_group << ", stop_group=" << stop_group;
																																				cout << "\n depth_disp_sum_reults = (";
																																				for (int k=0; k<4; k++){
																																						cout << ", " << depth_disp_sum_reults[k] ;
																																				}cout << ")";
																																				cout << endl;

																																			}
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::measureDepthFit()_chk4_Finished"<<flush;
}


//////////////////////////////  New Primal-only kernels, for combining SIRFS-like and DTAM cost functions



void RunCL::SpatialCostFns(){
	int local_verbosity_threshold = verbosity_mp["RunCL::SpatialCostFns"];

}

void RunCL::ParsimonyCostFns(){
	int local_verbosity_threshold = verbosity_mp["RunCL::ParsimonyCostFns"];

}

void RunCL::ExhaustiveSearch(){
	int local_verbosity_threshold = verbosity_mp["RunCL::ExhaustiveSearch"];

}




/* NB need separate OpenCL command queues for tracking, mapping, and data upload.
 * see "OpenCL_ Hide data transfer behind GPU Kernels runtime _ by Ravi Kumar _ Medium.mhtml"
 * NB occupancy of the GPU. Need to view with nvprof & Radeon™ GPU Profiler
 *
 * Initially just get the algorithm to work, then optimise data flows, GPU occupancy etc.
 */

// # create device buffer for image array[6]. NB MipMap needs 1.5 ximage size
// See CostVol::CostVol(..), RunCL::RunCL(..) & RunCL::allocatemem(..)

// from allocatemem(..) adapt for mipmapbuf
/*
dmem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE , width * height * sizeof(float), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}

status = clEnqueueWriteBuffer(uload_queue, gxmem, 		CL_FALSE, 0, width*height*sizeof(float), 		gx, 			0, NULL, &writeEvt);
    if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl; exit_(status);}

	cvrc.params[ROWS] 			= rows;
	cvrc.params[COLS] 			= cols;
*/


