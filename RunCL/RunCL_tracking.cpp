#include "RunCL.h"

void RunCL::precom_param_maps(float SE3_k2k[6*16]){ //  Compute maps of pixel motion for each SE3 DoF, and camera params // Derived from RunCL::mipmap
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::precom_param_maps(float SE3_k2k[6*16])_chk_0 "<<flush;}
	cl_event 			writeEvt;
	cl_int 				res, status;
	cv::Mat depth		= cv::Mat::ones (mm_height, mm_width, CV_32FC1);																	// NB must recompute translation maps at run time. NB parallax motion is proportional to inv depth.
	float mid_depth 	= (fp32_params[MAX_INV_DEPTH] + fp32_params[MIN_INV_DEPTH])/2.0;                                                    // TODO fix : depthmap not used as a kernel arg. NB want to match scale of depth range, but ? parallax may vary.
	depth 				*= mid_depth;
    //cout << "\n\ndepth.size()="<<depth.size()<<",  depth.type()="<< checkCVtype( depth.type() )  <<",  mid_depth="<<mid_depth<<endl<<flush;
	//cv::imshow("depth",depth);
    //cv::imwrite("/home/hockingsn/Programming/OpenCV/MySLAM/output/depth_precom_param_maps.tiff",depth);
    //cv::waitKey(-1);
	// SO3_k2kbuf
	status = clEnqueueWriteBuffer(uload_queue, SE3_k2kbuf, 		CL_FALSE, 0, 6*16*sizeof(float), SE3_k2k,    0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, depth_mem, 		CL_FALSE, 0, mm_size_bytes_C1,	 depth.data, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clFlush( uload_queue );																								if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	clFinish( uload_queue );
	//waitForEventAndRelease( &writeEvt );

	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	 uint	    layer,		//0
    res = clSetKernelArg(comp_param_maps_kernel, 1, sizeof(cl_mem),     &mipmap_buf);				            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*	mipmap_params,	//1
	res = clSetKernelArg(comp_param_maps_kernel, 2, sizeof(cl_mem), 	&uint_param_buf);	                    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	uint*	uint_params		//2
	res = clSetKernelArg(comp_param_maps_kernel, 3, sizeof(cl_mem), 	&SE3_k2kbuf);		                    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float* 	k2k,			//3
	res = clSetKernelArg(comp_param_maps_kernel, 4, sizeof(cl_mem), 	&SE3_map_mem);		                    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float* 	SE3_map,		//4
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precom_param_maps(float SO3_k2k[6*16])_chk_1 "<<flush;}
	// SE3_map_mem, k_map_mem, dist_map_mem;
	mipmap_call_kernel( comp_param_maps_kernel, m_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precom_param_maps(float SO3_k2k[6*16])_chk_2 "<<flush;}
																																			if(verbosity>local_verbosity_threshold) {
																																																cout<<"\n\nRunCL::precom_param_maps(float SO3_k2k[6*16])_output "<<flush;
																																																for (int i=0; i<1; i++) { // TODO x & y for all 6 SE3 DoF
																																																	stringstream ss;	ss << dataset_frame_num << "_SE3_map";
																																																	DownloadAndSave_2Channel_volume(SE3_map_mem, ss.str(), paths.at("SE3_map_mem"), mm_size_bytes_C1*2, mm_Image_size, CV_32FC2, false, 1.0, 6 /*SE3, 6DoF */);
																																																}
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precom_param_maps(float SE3_k2k[6*16])_chk.. Finished "<<flush;}
}

void RunCL::se3_rho_sq(float Rho_sq_results[8][4], const float count[4], uint start, uint stop ){
	int local_verbosity_threshold = -3;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk0 .##################################################################"<<flush;}
	cl_event writeEvt;
	cl_int status;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::se3_rho_sq(..)__chk_0.3: K2K= ";
																																				for (int i=0; i<16; i++){ cout << ",  "<< fp32_k2k[i];  }	cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk0.4 ,  dataset_frame_num="<<dataset_frame_num<<",   count="<<count[0]<<flush;}

	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,	CL_FALSE, 0, 16 * sizeof(float), fp32_k2k, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
/*	// Zero the output buffers.
	__global	float4* Rho_,					//8
	__global 	float4*	global_sum_rho_sq		//10
*/
//	status = clEnqueueFillBuffer(uload_queue, gxmem, 	&zero, sizeof(float), 0, mm_size_bytes_C4, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	float zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, SE3_rho_map_mem, 		&zero, sizeof(float), 0, 2*mm_size_bytes_C4,	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, se3_sum_rho_sq_mem, 	&zero, sizeof(float), 0, pix_sum_size_bytes, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.9\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clFlush( uload_queue );																						if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	clFinish( uload_queue );
																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk0.7 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData(..)
	cl_int 				res;
	//input
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	    uint	    layer,		                    //0
    res = clSetKernelArg(se3_rho_sq_kernel, 1, sizeof(cl_mem), &mipmap_buf);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	                //1
	res = clSetKernelArg(se3_rho_sq_kernel, 2, sizeof(cl_mem), &uint_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(se3_rho_sq_kernel, 3, sizeof(cl_mem), &fp32_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//3
	res = clSetKernelArg(se3_rho_sq_kernel, 4, sizeof(cl_mem), &k2kbuf);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//4		// TODO keyframe2K
	res = clSetKernelArg(se3_rho_sq_kernel, 5, sizeof(cl_mem), &keyframe_imgmem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//5		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	res = clSetKernelArg(se3_rho_sq_kernel, 6, sizeof(cl_mem), &imgmem);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem,							//6
	res = clSetKernelArg(se3_rho_sq_kernel, 7, sizeof(cl_mem), &keyframe_depth_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float*		keyframe_depth_mem				//7		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	//output
	res = clSetKernelArg(se3_rho_sq_kernel, 8, sizeof(cl_mem), &SE3_rho_map_mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_					        //8
	res = clSetKernelArg(se3_rho_sq_kernel, 9, local_work_size*4*sizeof(float), NULL);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_rho_sq				//9		// 1 DoF, float4 channels
	res = clSetKernelArg(se3_rho_sq_kernel,10, sizeof(cl_mem), &se3_sum_rho_sq_mem);		 						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_rho_sq,				//10

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk1 ."<<flush;}
	mipmap_call_kernel( se3_rho_sq_kernel, m_queue, start, stop );

																																			 if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num <<"_iter_"<<count[0]<<"_layer"<<count[1]<<"_factor"<<count[2]<<"_se3_rho_sq_";
																																				stringstream ss_path;

																																				bool display = false; //obj["sample_se3_incr"].asBool();

																																				cout << "\nRunCL::se3_rho_sq(..)_chk3.5   display="<< display<< endl << flush;
																																				DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at("SE3_rho_map_mem"),  mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 1, 0, count[0], display );
																																			}


																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk4 ."<<flush;}
	cv::Mat rho_sq_sum_mat = cv::Mat::zeros (se3_sum_size, 4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)					// NB the data returned is one float4 per group, holding HSV, plus entry[3]=pixel count.
	ReadOutput( rho_sq_sum_mat.data, se3_sum_rho_sq_mem, pix_sum_size_bytes );																//float Rho_sq_reults[8][4] = {{0}};
																																			if(verbosity>local_verbosity_threshold+2) {
// DownloadAndSave_3Channel(buffer, ss.str(), folder, image_size_bytes, size_mat, type_mat, show, max_range, i*image_size_bytes, exception_tiff);
// ReadOutput(temp_mat.data, buffer,  image_size_bytes,   offset);
																																				cout << "\n\nRunCL::se3_rho_sq(..)_chk5 ."<<flush;
																																				cout << "\nrho_sq_sum_mat.size()="<<rho_sq_sum_mat.size()<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																				cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
																																				cout << "\n\nrho_sq_sum_mat.at<float> (i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 (H,S,V, (valid pixels/group_size) )";
																																				for (int i=0; i< se3_sum_size ; i++){//&& i<30
																																					cout << "\ngroup ="<<i<<":   ";
																																					cout << ",     \t(";
																																					for (int k=0; k<4; k++){	cout << ", \t" << rho_sq_sum_mat.at<float>(i, k); }
																																					cout << ")";
																																				}cout << endl << endl;
																																			}

	for (int i=0; i<=mm_num_reductions+1; i++){
		uint read_offset_ 			= MipMap[i*8 +MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
		uint global_sum_offset 		= read_offset_ / local_work_size ;
		uint groups_to_sum 			= rho_sq_sum_mat.at<float>(global_sum_offset, 0);
		uint start_group 			= global_sum_offset + 1;
		uint stop_group 			= start_group + groups_to_sum ;   																		// -1
																																			/*
																																			cout << "\nRunCL::se3_rho_sq(..)_chk6 layer = "<<i<<
																																			", read_offset_="<<read_offset_<<
																																			", global_sum_offset = "<<global_sum_offset<<
																																			", groups_to_sum = "<<groups_to_sum<<
																																			", start_group = "<<start_group<<
																																			", stop_group = "<<stop_group<< flush;
																																			*/
		for (int j=start_group; j< stop_group; j++){	for (int l=0; l<4; l++){ 	Rho_sq_results[i][l] += rho_sq_sum_mat.at<float>(j, l);		};
			//cout << "\n #group = " << j << ", Value = " << rho_sq_sum_mat.at<float>(j, 2);
		}																									// sum j groups for this layer of the MipMap.
	}
																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << endl;
																																				for (int i=0; i<=mm_num_reductions+1; i++){ 																// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<i<<" mm_num_reductions = "<< mm_num_reductions <<",  Rho_sq_results/num_groups = (";
																																					if (Rho_sq_results[i][3] > 0){
																																						for (int l=0; l<3; l++){	cout << ", \t" << Rho_sq_results[i][l] / ( Rho_sq_results[i][3]  *  img_stats[IMG_VAR+l]  );
																																						}
																																						cout << ", \t" << Rho_sq_results[i][3] << ")";
																																					}
																																					else{	for (int l=0; l<3; l++){	cout << ", \t" << 0.0f;		}
																																						cout << ", \t" << Rho_sq_results[i][3] << ")";
																																					}
																																				}cout << "\n\nRunCL::se3_rho_sq(..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																			}
}

// TODO   Move writeToResultsMat(..) into DownloadAndSave_3Channel_volume   AND pass back a Mat from DownloadAndSave_3Channel   // Question : why is reading the buffer twice such a problem ?  // why is it crashing ?
void RunCL::writeToResultsMat(cv::Mat *bufImg , uint column_of_images , uint row_of_images ){													// writeToResultsMat(buffer , column of images = iteration, row of images );
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk0,\t column_of_images="<<column_of_images<<"\t row_of_images="<<row_of_images<<" "<<flush;}
	uint reduction 			= obj["sample_layer"].asUInt();																					// extract patch
	//Mat temp_mat 			= Mat::zeros (mm_Image_size, CV_32FC4);																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk1"<<flush;}
	//ReadOutput(temp_mat.data, buffer,  image_size_bytes);																					if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk2"<<flush;}
/*
	cv::namedWindow( "writeToResultsMat" , 0 );
	cv::imshow("writeToResultsMat", *bufImg);
	cv::waitKey(-1);
	destroyWindow( "writeToResultsMat" );
*/
	int patch_rows 			= MipMap[reduction*8 + MiM_READ_ROWS];
	int patch_cols 			= MipMap[reduction*8 + MiM_READ_COLS];
	int row_offset 			= mm_margin * reduction;																						if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk3"<<flush;}
	for (int layer = 0; layer < reduction; layer ++)  { row_offset += MipMap[layer*8 + MiM_READ_ROWS]; }									if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk4"<<flush;}
	int col_offset 			= mm_margin ;																									if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk5"<<flush;}
	int row_offset2			= mm_margin + row_of_images * ( mm_margin + patch_rows );																// paste patch
	int col_offset2			= mm_margin + column_of_images * ( mm_margin + patch_cols );													if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk5.1"<<flush;}
	/*
	cv::namedWindow( "writeToResultsMat_patch" , 0 );
	Mat temp_mat;
	temp_mat.data = bufImg->data;
	cv::imshow("writeToResultsMat_patch", temp_mat(cv::Rect( col_offset, row_offset, patch_cols, patch_rows )));
	cv::waitKey(-1);
	destroyWindow( "writeToResultsMat_patch" );
	*/

	//cout << "\nRunCL::writeToResultsMat(..)  resultsMat.type() = " << resultsMat.type()  << "  " <<  checkCVtype(resultsMat.type())  << ",    bufImg->type() = " << bufImg->type()  << "  " <<  checkCVtype(bufImg->type())  << endl << flush;

	for (int col = 0; col < patch_cols ; col++ ) {
		for (int row = 0; row < patch_rows ; row ++) {
			resultsMat.at<Vec4b>(row+row_offset2 , col+col_offset2) = bufImg->at<Vec4b>(row+row_offset , col+col_offset);  // wrong vec type for img
		}
	}																																		if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk finished"<<flush;}
	/*
	cv::namedWindow( "writeToResultsMat: resultsMat" , 0 );
	cv::imshow("writeToResultsMat: resultsMat", resultsMat);
	cv::waitKey(-1);
	destroyWindow( "writeToResultsMat: resultsMat" );
	*/
}


void RunCL::estimateSO3(float SO3_results[8][3][4], float Rho_sq_results[8][4], int count, uint start, uint stop){ //estimateSO3();	(uint start=0, uint stop=8)
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk0 . #######################################################################"<<flush;}
    cl_event writeEvt;
    cl_int status;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::estimateSO3(..)__chk_0.5: fp32_so3= ";
																																				for (int i=0; i<9; i++){ cout << ",  "<< fp32_so3_k2k[i]; }cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk0.6 ,  dataset_frame_num="<<dataset_frame_num<<",   count="<<count<<flush;}
																																			// SO3_k2kbuf, fp32_so3_k2k
	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,	CL_FALSE, 0, 16 * sizeof(float), fp32_k2k, 	0, NULL, &writeEvt);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSO3(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	// TODO zero buffers, as in estimateSE3(..) OR eliminate SO3.

	status = clFlush( uload_queue );																								if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	waitForEventAndRelease( &writeEvt );
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk1 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData(..)
	cl_int 				res;
	// inputs:
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	    uint	    layer,		                    //0
    res = clSetKernelArg(so3_grad_kernel, 1, sizeof(cl_mem), &mipmap_buf);				                        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	                //1
	res = clSetKernelArg(so3_grad_kernel, 2, sizeof(cl_mem), &uint_param_buf);						            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(so3_grad_kernel, 3, sizeof(cl_mem), &fp32_param_buf);						            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//3
	res = clSetKernelArg(so3_grad_kernel, 4, sizeof(cl_mem), &k2kbuf);								        	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//4		// TODO keyframe2K
	res = clSetKernelArg(so3_grad_kernel, 5, sizeof(cl_mem), &keyframe_imgmem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//5		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	res = clSetKernelArg(so3_grad_kernel, 6, sizeof(cl_mem), &imgmem);				            				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem	,						//6
	res = clSetKernelArg(so3_grad_kernel, 7, sizeof(cl_mem), &keyframe_SE3_grad_map_mem);	            		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		keyframe_SE3_grad_map_mem)		//7
	res = clSetKernelArg(so3_grad_kernel, 8, sizeof(cl_mem), &SE3_grad_map_mem);	            				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_grad_map					//8
	// outputs:
	res = clSetKernelArg(so3_grad_kernel, 9, local_work_size*3*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_grads					//9		3 DoF, float4 channels TODO why 7 ?
	res = clSetKernelArg(so3_grad_kernel,10, sizeof(cl_mem), &se3_sum_mem);		 					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_grads,				//10
	res = clSetKernelArg(so3_grad_kernel,11, sizeof(cl_mem), &SE3_incr_map_mem);					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_incr_map_					//11
	res = clSetKernelArg(so3_grad_kernel,12, sizeof(cl_mem), &SE3_rho_map_mem);					                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_					        //12
	res = clSetKernelArg(so3_grad_kernel,13, local_work_size*8*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_rho_sq				//13	1 DoF, float4 channels
	res = clSetKernelArg(so3_grad_kernel,14, sizeof(cl_mem), &se3_sum_rho_sq_mem);		 					    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_rho_sq,				//14

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk2 ."<<flush;}
	mipmap_call_kernel( so3_grad_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_iter_"<< count << "_estimateSO3_";
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_incr_map_mem, ss.str(), paths.at("SO3_incr_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6, false ); //false
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at("SO3_rho_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 1, false );  //false
/*
																																				//DownloadAndSave_3Channel_volume(  keyframe_imgmem,  ss.str(), paths.at("keyframe_imgmem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, 1, 1 );
																																				//DownloadAndSave_3Channel_volume(  imgmem,  ss.str(), paths.at("imgmem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, 1, 1 );
*/
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk4 ."<<flush;}		// directly read higher layers
	const uint num_DoFs = 3;
    cv::Mat so3_sum_mat = cv::Mat::zeros (so3_sum_size, num_DoFs*4, CV_32FC1); 		// cv::Mat::zeros (int rows, int cols, int type)		// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( so3_sum_mat.data, se3_sum_mem, so3_sum_size_bytes );                                                                        // so3_sum_size = num rows = (num work_groups + padding), in se3_sum_mem, when used for "global_sum_grads" in  __kernel void so3_grad(...) .
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n\nRunCL::estimateSO3(..)_chk5 ."	<<flush;
																																				cout << "\nso3_sum_mat.size()="				<<so3_sum_mat.size()	<<flush;
																																				cout << "\nso3_sum_size="					<<so3_sum_size			<<flush;
                                                                                                                                                cout << "\nmm_num_reductions = " 			<< mm_num_reductions 	<<flush;
																																				cout << "\nso3_sum_mat.at<float> (i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 (H,S,V, (valid pixels/group_size) )";
																																				for (int i=0; i< so3_sum_size ; i++){
																																					cout << "\ngroup ="<<i<<":   ";
																																					for (int j=0; j<num_DoFs; j++){
																																						cout << ",     \t(";	for (int k=0; k<4; k++){	cout << ", \t" << so3_sum_mat.at<float> (i , j*4 + k); } 	cout << ")";
																																					}cout << flush;
																																				}
																																				cout << endl << endl;
																																			}// if start, stop  larger layers, call reduce kernel. ? cut off between large vs small layers ?
	for (int i=0; i<=mm_num_reductions; i++){																																				//######################################################## // For (each reduction layer) of the image pyramid.
        uint read_offset_ 		= MipMap[i*8 + MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
        uint global_sum_offset 	= read_offset_ / local_work_size ;
        uint groups_to_sum 		= so3_sum_mat.at<float>(global_sum_offset, 0);
        uint start_group 		= global_sum_offset + 1;
        uint stop_group 		= start_group + groups_to_sum ;																				// skip the last group due to odd 7th value.
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\ni="<<i<<", read_offset_="<<read_offset_<<",  global_sum_offset="<<global_sum_offset<<",  groups_to_sum="<<groups_to_sum<< ",  start_group="<<start_group<<",  stop_group="<<stop_group;
																																			}
        for (int j=start_group; j< stop_group  ; j++){ 	for (int k=0; k<num_DoFs; k++){ 	for (int l=0; l<4; l++){	SO3_results[i][k][l] += so3_sum_mat.at<float>(j, k*4 + l);	} }	}	//#########################################################	// Compute global sum of SO3 delta4 for each SO3 DoF.
    }																																														//######################################################### // end: For (each reduction layer) loop
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n\nSO3_results[reduction layer][SO3_DoF][delta4(H,S,V,count)]";
																																				for (int i=0; i<=mm_num_reductions+1; i++){ 															// results
																																					cout << "\nLayer "<<i<<" SE3_results = (";																// Raw results, for each reduction layer : Display as text
																																					for (int k=0; k<num_DoFs; k++){																			// for each SO3 DoF
																																						cout << "\t(";
																																						for (int l=0; l<4; l++){																			// for each HSV colour channel.
																																							cout << ", \t" << SO3_results[i][k][l] ;														// i=reduction layer, k=SO3 DoF, l=delta4 (H,S,V, (valid pixels/group_size) )
																																						}cout << ")";
																																					}cout << ")";
																																				}
																																				cout << endl;
																																				for (int i=0; i<=mm_num_reductions+1; i++){ 															// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<i<<" SO3_results/((num valid pixels/group size) * image variance) = (";
																																					for (int k=0; k<num_DoFs; k++){
																																						cout << "\t(";
																																						for (int l=0; l<3; l++){
																																							cout << ", \t" << SO3_results[i][k][l] / (SO3_results[i][k][3]  *  img_stats[IMG_VAR+l]  );		// divide by ((num valid pixels/group size) * the whole image variance)
																																						}
																																						cout << ", " << SO3_results[i][k][3] << ")";
																																					}cout << ")";
																																				}
																																			}

    cv::Mat rho_sq_sum_mat = cv::Mat::zeros (so3_sum_size, 4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)					// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( rho_sq_sum_mat.data, se3_sum_rho_sq_mem, pix_sum_size_bytes );

	//float Rho_sq_results[8][4] = {{0}};
	for (int i=0; i<=mm_num_reductions; i++){
		uint read_offset_ 			= MipMap[i*8 +MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
		uint global_sum_offset 		= read_offset_ / local_work_size ;
		uint groups_to_sum 			= so3_sum_mat.at<float>(global_sum_offset, 0);
		uint start_group 			= global_sum_offset + 1;
		uint stop_group 			= start_group + groups_to_sum ;   // -1

		for (int j=start_group; j< stop_group; j++){
			for (int l=0; l<4; l++){
				Rho_sq_results[i][l] += rho_sq_sum_mat.at<float>(j, l);																		// sum j groups for this layer of the MipMap.
			};
		}
	}
																																			if(verbosity>local_verbosity_threshold ) {
																																				cout << endl;
																																				for (int i=0; i<=mm_num_reductions+1; i++){ 															// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<i<<" mm_num_reductions = "<< mm_num_reductions <<",  Rho_sq_results/num_groups = (";

																																					if (Rho_sq_results[i][3] > 0){
																																						for (int l=0; l<3; l++){
																																							cout << ", " << Rho_sq_results[i][l] / ( Rho_sq_results[i][3]  *  img_stats[IMG_VAR+l]  );	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}
																																						cout << ", " << Rho_sq_results[i][3] << ")";
																																					}
																																					else{
																																						for (int l=0; l<3; l++){
																																							cout << ", " << 0.0f  ;	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}
																																						cout << ", " << Rho_sq_results[i][3] << ")";
																																					}
																																				}
																																			}

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_finished . +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;}
}

void RunCL::estimateSE3(float SE3_results[8][6][tracking_num_colour_channels], float Rho_sq_results[8][4], int count, uint start, uint stop){ //estimateSE3(); 	(uint start=0, uint stop=8)			// TODO replace arbitrary fixed constant with a const uint variable in the header...
	int local_verbosity_threshold = -2;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0 .##################################################################"<<flush;}
    cl_event writeEvt;
    cl_int status;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::estimateSE3(..)__chk_0.3: K2K= ";
																																				for (int i=0; i<16; i++){ cout << ",  "<< fp32_k2k[i];  }	cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0.4 ,  dataset_frame_num="<<dataset_frame_num<<",   count="<<count<<flush;}

	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,	CL_FALSE, 0, 16 * sizeof(float), fp32_k2k, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSE3(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
/*	// Zero the output buffers.
	__global	float4*	global_sum_grads,		//11
	__global 	float4*	SE3_incr_map_,			//12
*/
//	status = clEnqueueFillBuffer(uload_queue, gxmem, 	&zero, sizeof(float), 0, mm_size_bytes_C4, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	float zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, se3_sum_mem, 			&zero, sizeof(float), 0, se3_sum_size_bytes, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSE3(..)_chk0.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, SE3_incr_map_mem, 	&zero, sizeof(float), 0, mm_size_bytes_C1*24, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSE3(..)_chk0.7\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clFlush( uload_queue );																								if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	clFinish( uload_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0.7 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData(..)
	cl_int 				res;
																																																	//input
	//      __private	 uint layer, set in mipmap_call_kernel(..) below																															__private		uint		layer,							//0
	res = clSetKernelArg(se3_grad_kernel, 1, sizeof(cl_mem), &mipmap_buf);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		mipmap_params,					//1
	res = clSetKernelArg(se3_grad_kernel, 2, sizeof(cl_mem), &uint_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(se3_grad_kernel, 3, sizeof(cl_mem), &fp32_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//3
	res = clSetKernelArg(se3_grad_kernel, 4, sizeof(cl_mem), &k2kbuf);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//4		// TODO keyframe2K
	res = clSetKernelArg(se3_grad_kernel, 5, sizeof(cl_mem), &keyframe_imgmem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//5		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	res = clSetKernelArg(se3_grad_kernel, 6, sizeof(cl_mem), &imgmem);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem,							//6
	res = clSetKernelArg(se3_grad_kernel, 7, sizeof(cl_mem), &keyframe_SE3_grad_map_mem);						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		keyframe_SE3_grad_map_mem		//7
	res = clSetKernelArg(se3_grad_kernel, 8, sizeof(cl_mem), &SE3_grad_map_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_grad_map					//8
	res = clSetKernelArg(se3_grad_kernel, 9, sizeof(cl_mem), &keyframe_depth_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float*		keyframe_depth_mem				//9		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	res = clSetKernelArg(se3_grad_kernel,10, sizeof(cl_mem), &SE3_rho_map_mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_							//10
																																																	//output
	res = clSetKernelArg(se3_grad_kernel,11, local_work_size*6*4*sizeof(float), NULL);							if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_grads					//11	6 DoF, float4 channels
	res = clSetKernelArg(se3_grad_kernel,12, sizeof(cl_mem), &se3_sum_mem);		 								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_grads,				//12
	res = clSetKernelArg(se3_grad_kernel,13, sizeof(cl_mem), &SE3_incr_map_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_incr_map_					//13

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk1 ."<<flush;}
	mipmap_call_kernel( se3_grad_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_iter_"<< count << "_estimateSE3_";
                                                                                                                                                stringstream ss_path;

																																				bool display = false; //obj["sample_se3_incr"].asBool();
																																				DownloadAndSave_3Channel_volume(  SE3_incr_map_mem, ss.str(), paths.at("SE3_incr_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6, 0, count, display );
																																				//DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at("SE3_rho_map_mem"),  mm_size_bytes_C4, mm_Image_size, CV_32FC4, true, -1, 1, 0, count, display );
																																			}
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk4 ."<<flush;}
	//	mipmap_params set in mipmap_call_kernel(..) below																					                                                          __constant 	uint*		mipmap_params,	//0
	res = clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), &uint_param_buf);                                    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant 	uint*		uint_params,	//1
	res = clSetKernelArg(reduce_kernel, 2, sizeof(cl_mem), &se3_sum_mem);                                       if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float8*		se3_sum			//2
	res = clSetKernelArg(reduce_kernel, 3, local_work_size*4*sizeof(float), 	NULL);							if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local    	float8*		local_sum_grads	//3
	res = clSetKernelArg(reduce_kernel, 4, sizeof(cl_mem), &se3_sum2_mem);                                      if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float8*		se3_sum2,		//4
// TODO implement & use reduce_kernel for 1st 2 layers. 
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk5 ."<<flush;}
                                                                                                                                            // directly read higher layers
	uint num_DoFs = 6;
    cv::Mat se3_sum_mat = cv::Mat::zeros (se3_sum_size, num_DoFs*4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)				// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( se3_sum_mat.data, se3_sum_mem, se3_sum_size_bytes );                                                                        // se3_sum_size_bytes
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\n\nRunCL::estimateSE3(..)_chk6 ."<<flush;
																																				cout << "\nse3_sum_mat.size()="<<se3_sum_mat.size()<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																				cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
																																				cout << "\n\nse3_sum_mat.at<float> (i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 (H,S,V, (valid pixels/group_size) )";
																																				for (int i=0; i< se3_sum_size ; i++){//&& i<30
																																					cout << "\ngroup ="<<i<<":   ";
																																					for (int j=0; j<num_DoFs; j++){
																																						cout << ",     \t(";	for (int k=0; k<4; k++){	cout << ", \t" << se3_sum_mat.at<float>(i, j*4 + k); }	cout << ")";
																																					}cout << flush;
																																				}cout << endl << endl;
																																			}
	for (int i=0; i<=mm_num_reductions+1; i++){
        uint read_offset_ 		= MipMap[i*8 + MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
        uint global_sum_offset 	= read_offset_ / local_work_size ;
        uint groups_to_sum 		= se3_sum_mat.at<float>(global_sum_offset, 0);
        uint start_group 		= global_sum_offset + 1;
        uint stop_group 		= start_group + groups_to_sum ;   // -1																		// skip the last group due to odd 7th value.
																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << "\ni="<<i<<", read_offset_="<<read_offset_<<",  global_sum_offset="<<global_sum_offset<<",  groups_to_sum="<<groups_to_sum<< ",  start_group="<<start_group<<",  stop_group="<<stop_group;
																																			}
		for (int j=start_group; j< stop_group  ; j++){	for (int k=0; k<num_DoFs; k++){ 	for (int l=0; l<4; l++){	SE3_results[i][k][l] += se3_sum_mat.at<float>(j, k*4 + l);	} }	}			//l =4 =num channels	// sum j groups for this layer of the MipMap. // se3_sum_mat.at<float>(j, k);
    }

																																		//	if (obj["sample_se3_incr"].asBool() == true){
																																		//		writeToResultsMat(SE3_rho_map_mem, count[0], 0 );	// writeToResultsMat(buffer , column of images = iteration, row of images );
																																		//	}

																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << endl << " SE3_results/num_groups = (H, S, V, alpha=num_groups) ";
																																				for (int i=0; i<=mm_num_reductions+1; i++){ 																// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<i<<" SE3_results = (";																	// raw results
																																					for (int k=0; k<num_DoFs; k++){
																																						cout << "\n(";  for (int l=0; l<4; l++){	cout << ", \t" << SE3_results[i][k][l] ;	}cout << ")";
																																					}cout << ")";
																																					///
																																					cout << "\nLayer "<<i<<" SE3_results/num_groups = (";
																																					for (int k=0; k<num_DoFs; k++){
																																						cout << "\nDoF="<<k<<" ("; for (int l=0; l<3; l++){	cout << ", \t" << SE3_results[i][k][l] / ( SE3_results[i][k][3]  *  img_stats[IMG_VAR+l]  ); } cout << ", " << SE3_results[i][k][3] << ")";
																																					}cout << ")";																							// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																				}cout << "\n\nRunCL::estimateSE3(..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																			}
	/*
	cv::Mat rho_sq_sum_mat = cv::Mat::zeros (se3_sum_size, 4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)					// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( rho_sq_sum_mat.data, se3_sum_rho_sq_mem, pix_sum_size_bytes );																//float Rho_sq_reults[8][4] = {{0}};

	for (int i=0; i<=mm_num_reductions+1; i++){
		uint read_offset_ 			= MipMap[i*8 +MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
		uint global_sum_offset 		= read_offset_ / local_work_size ;
		uint groups_to_sum 			= se3_sum_mat.at<float>(global_sum_offset, 0);
		uint start_group 			= global_sum_offset + 1;
		uint stop_group 			= start_group + groups_to_sum ;   																		// -1
		for (int j=start_group; j< stop_group; j++){	for (int l=0; l<4; l++){ 	Rho_sq_results[i][l] += rho_sq_sum_mat.at<float>(j, l);		}; 		}																									// sum j groups for this layer of the MipMap.
	}
																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << endl;
																																				for (int i=0; i<=mm_num_reductions+1; i++){ 																// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<i<<" mm_num_reductions = "<< mm_num_reductions <<",  Rho_sq_results/num_groups = (";
																																					if (Rho_sq_results[i][3] > 0){
																																						for (int l=0; l<3; l++){	cout << ", \t" << Rho_sq_results[i][l] / ( Rho_sq_results[i][3]  *  img_stats[IMG_VAR+l]  );
																																						}
																																						cout << ", \t" << Rho_sq_results[i][3] << ")";
																																					}
																																					else{	for (int l=0; l<3; l++){	cout << ", \t" << 0.0f;		}
																																						cout << ", \t" << Rho_sq_results[i][3] << ")";
																																					}
																																				}
																																			}
	*/
}

void RunCL::tracking_result(string result){
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::tracking_result(..)_chk0"<<flush;
																																				stringstream ss;			ss << dataset_frame_num <<  "_img_grad_" << result;				// "_iter_"<< count <<
																																				stringstream ss_path_rho;	ss_path_rho << "SE3_rho_map_mem";
																																				cout << " , " << ss_path_rho.str() << " , " <<  paths.at(ss_path_rho.str()) << " , " << ss.str()  <<flush;
																																				DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at(ss_path_rho.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 1 );
																																			}
}


	/*
	 * atomic_test1_buf	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, 4*local_work_size*sizeof(int),	0, &res);	if(res!=CL_SUCCESS){cout<<"\nres 42= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	 */

void RunCL::atomic_test1(){
	cl_int res, status;
	cl_event ev, writeEvt;
	const int data_size = 4*local_work_size;
	const size_t num_threads = 2 * local_work_size;
	int num_threads_int = 1.5*local_work_size;

	int zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, atomic_test1_buf, 	&zero, sizeof(int), 0, data_size*sizeof(int), 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}

	clFlush(uload_queue); status = clFinish(uload_queue); 																					if (status != CL_SUCCESS)	{ cout << "\nclFinish(uload_queue)=" << status << checkerror(status) <<"\n"  << flush; exit_(status);}

	res = clSetKernelArg(atomic_test1_kernel, 0, sizeof(int), 		&num_threads_int);														if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;
	res = clSetKernelArg(atomic_test1_kernel, 1, sizeof(cl_mem), 	&atomic_test1_buf);														if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	/*
	cl_int clEnqueueNDRangeKernel(
									cl_command_queue 		command_queue,
									cl_kernel 				kernel,
									cl_uint 				work_dim,
									const size_t* 			global_work_offset,
									const size_t* 			global_work_size,
									const size_t* 			local_work_size,
									cl_uint 				num_events_in_wait_list,
									const cl_event* 		event_wait_list,
									cl_event* 				event
									);
	*/
	res 	= clEnqueueNDRangeKernel(m_queue, atomic_test1_kernel, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev);						if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(m_queue);																												if (status != CL_SUCCESS)	{ cout << "\nRunCL::atomic_test1(),  clFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}

	int atomic_test_output[ data_size ];
	for (int i = 0; i<data_size; i++) atomic_test_output[i] = -1;
	cl_event readEvt;
	status = clEnqueueReadBuffer(	dload_queue,			// command_queue
									atomic_test1_buf,		// buffer
									CL_FALSE,				// blocking_read
									0,						// offset
									data_size*sizeof(int),	// size
									atomic_test_output,		// pointer
									0,						// num_events_in_wait_list
									NULL,					// event_waitlist				needs to know about preceeding events:
									&readEvt				// event
								 );
													if (status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer(..) status=" << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFlush(dload_queue);					if (status != CL_SUCCESS) { cout << "\nclFlush(m_queue) status = " 		<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents(1, &readEvt); 			if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status="			<< checkerror(status) <<"\n"<<flush; exit_(status);}

	cout << "\n\n void RunCL::atomic_test1(): \t  local_work_size = "<<local_work_size<<", \t (buffer size) data_size = "<<data_size<<", \t (num threards launched)  num_threads="<<num_threads<<", \t (num threards run) num_threads_int = "<<num_threads_int<<", \t  (";
	int i=0;
	for (; i< data_size; i++) cout << ", " << atomic_test_output[i];
	cout << ") \t i="<< i<< "\n\n" << flush;
}



