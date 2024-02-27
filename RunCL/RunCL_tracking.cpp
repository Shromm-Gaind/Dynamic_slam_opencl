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

void RunCL::estimateSO3(float SO3_results[8][3][4], float Rho_sq_results[8][4], int count, uint start, uint stop){ //estimateSO3();	(uint start=0, uint stop=8)
	int local_verbosity_threshold = -2;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk0 ."<<flush;}
    cl_event writeEvt;
    cl_int status;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::estimateSO3(..)__chk_0.5: fp32_so3= ";
																																				for (int i=0; i<9; i++){ cout << ",  "<< fp32_so3_k2k[i]; }cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk0.6 ,  dataset_frame_num="<<dataset_frame_num<<",   count="<<count<<flush;}
																																			// SO3_k2kbuf, fp32_so3_k2k
	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,	CL_FALSE, 0, 16 * sizeof(float), fp32_k2k, 	0, NULL, &writeEvt);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSO3(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk1 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData(..)
	cl_int 				res;
	//size_t local_size = local_work_size;
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	    uint	    layer,		                    //0
    res = clSetKernelArg(so3_grad_kernel, 1, sizeof(cl_mem), &mipmap_buf);				                        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	                //1
	res = clSetKernelArg(so3_grad_kernel, 2, sizeof(cl_mem), &uint_param_buf);						            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(so3_grad_kernel, 3, sizeof(cl_mem), &fp32_param_buf);						            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//3
	res = clSetKernelArg(so3_grad_kernel, 4, sizeof(cl_mem), &k2kbuf);								        	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//4		// TODO keyframe2K
	res = clSetKernelArg(so3_grad_kernel, 5, sizeof(cl_mem), &keyframe_imgmem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//5		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	res = clSetKernelArg(so3_grad_kernel, 6, sizeof(cl_mem), &imgmem);				            				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem	,						//6
	res = clSetKernelArg(so3_grad_kernel, 7, sizeof(cl_mem), &keyframe_SE3_grad_map_mem);	            		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		keyframe_SE3_grad_map_mem)		//7
	res = clSetKernelArg(so3_grad_kernel, 8, sizeof(cl_mem), &SE3_grad_map_mem);	            				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_grad_map					//8
	//res = clSetKernelArg(so3_grad_kernel, 9, sizeof(cl_mem), &depth_mem);						                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		depth_map					    //9		// NB GT_depth, not inv_depth depth_mem
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

                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_incr_map_mem, ss.str(), paths.at("SO3_incr_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 ); //false

																																				cout<<"\n\nRunCL::estimateSO3(..)_chk3.6 ."<<flush;
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at("SO3_rho_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 1 );  //false

																																				//cout<<"\n\nRunCL::estimateSO3(..)_chk3.7 ."<<flush;
																																				//DownloadAndSave_3Channel_volume(  keyframe_imgmem,  ss.str(), paths.at("keyframe_imgmem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, 1, 1 );

																																				//cout<<"\n\nRunCL::estimateSO3(..)_chk3.8 ."<<flush;
																																				//DownloadAndSave_3Channel_volume(  imgmem,  ss.str(), paths.at("imgmem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, 1, 1 );
																																			}
																																			if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::estimateSO3(..)_chk5 ."<<flush;}
                                                                                                                                            // directly read higher layers
	uint num_DoFs = 3;

    cv::Mat so3_sum_mat = cv::Mat::zeros (so3_sum_size, num_DoFs*4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)				// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( so3_sum_mat.data, se3_sum_mem, so3_sum_size_bytes );                                                                        // so3_sum_size_bytes
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\nso3_sum_mat.size()="<<so3_sum_mat.size()<<flush;
																																				cout << "\nso3_sum_size="<<so3_sum_size<<flush;
																																			}
                                                                                                                                            if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::estimateSO3(..)_chk5 ."<<flush;
                                                                                                                                                cout << "\n\n so3_sum_mat.data = (\n";
                                                                                                                                                for (int i=0; i<so3_sum_size; i++){
                                                                                                                                                    cout << "\n group="<<i<<" : ( " << flush;
                                                                                                                                                    for (int j=0; j<8; j++){
                                                                                                                                                                                                    //cout << ",  \nso3_sum_mat.at<float>("<<i<<","<<j<<") = " << flush;
                                                                                                                                                        cout << so3_sum_mat.at<float>(i,j) << " , " << flush;
                                                                                                                                                    }
                                                                                                                                                    cout << ")" << flush;
                                                                                                                                                }cout << "\n)\n" << flush;
                                                                                                                                                cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
                                                                                                                                            }

                                                                                                                                            // if start, stop  larger layers, call reduce kernel. ? cut off between large vs small layers ?
    //float SO3_reults[8][6][4] = {{{0}}}; 																									// max 8 layers, 6+1 DoF, 4 channels
																																			if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::estimateSO3(..)_chk6 ."<<flush;
																																				cout << "\n\nso3_sum_mat.at<float> (i*num_DoFs + j,  k) ";
																																				for (int i=0; i< so3_sum_size ; i++){
																																					cout << "\ni="<<i<<":   ";
																																					for (int j=0; j<num_DoFs; j++){
																																						cout << ",     (";
																																						for (int k=0; k<4; k++){
																																							cout << "," << so3_sum_mat.at<float> (i , j*4 + k)  ;
																																						}cout << ")";
																																					}cout << flush;
																																				}
																																				cout << endl << endl;
																																			}

	for (int i=0; i<=mm_num_reductions+1; i++){
        uint read_offset_ 		= MipMap[i*8 + MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
        uint global_sum_offset 	= read_offset_ / local_work_size ;
        uint groups_to_sum 		= so3_sum_mat.at<float>(global_sum_offset, 0);
        uint start_group 		= global_sum_offset + 1;
        uint stop_group 		= start_group + groups_to_sum ;   // -1																		// skip the last group due to odd 7th value.
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\ni="<<i<<", read_offset_="<<read_offset_<<",  global_sum_offset="<<global_sum_offset<<",  groups_to_sum="<<groups_to_sum<< ",  start_group="<<start_group<<",  stop_group="<<stop_group;
																																			}
        for (int j=start_group; j< stop_group  ; j++){
            for (int k=0; k<num_DoFs; k++){
				for (int l=0; l<4; l++){
					SO3_results[i][k][l] += so3_sum_mat.at<float>(j, k*4 + l); // so3_sum_mat.at<float>(j, k);                         		// sum j groups for this layer of the MipMap.
				}
            }
        }
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\nLayer "<<i<<" SE3_results = (";																// raw results
																																				for (int k=0; k<num_DoFs; k++){
																																					cout << "(";
																																					for (int l=0; l<4; l++){
																																						cout << ", " << SO3_results[i][k][l] ;
																																					}cout << ")";
																																				}cout << ")";
																																			}
    }
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << endl;
																																				for (int i=0; i<=mm_num_reductions+1; i++){ 															// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<i<<" SE3_results/num_groups = (";
																																					for (int k=0; k<num_DoFs; k++){
																																						cout << "(";
																																						for (int l=0; l<3; l++){
																																							cout << ", " << SO3_results[i][k][l] / (SO3_results[i][k][3]  *  img_stats[IMG_VAR+l]  );	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}
																																						cout << ", " << SO3_results[i][k][3] << ")";
																																					}cout << ")";
																																				}
																																			}

    cv::Mat rho_sq_sum_mat = cv::Mat::zeros (so3_sum_size, 4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)					// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( rho_sq_sum_mat.data, se3_sum_rho_sq_mem, pix_sum_size_bytes );

	//float Rho_sq_results[8][4] = {{0}};
	for (int i=0; i<=mm_num_reductions+1; i++){
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

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_finished ."<<flush;}
}

void RunCL::estimateSE3(float SE3_results[8][6][tracking_num_colour_channels], float Rho_sq_results[8][4], int count, uint start, uint stop){ //estimateSE3(); 	(uint start=0, uint stop=8)			// TODO replace arbitrary fixed constant with a const uint variable in the header...
	int local_verbosity_threshold = -2;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0 ."<<flush;}
    cl_event writeEvt;
    cl_int status;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::estimateSE3(..)__chk_0.5: K2K= ";
																																				for (int i=0; i<16; i++){
																																					cout << ",  "<< fp32_k2k[i]; // K2K ("<<i%4 <<","<< i/4<<") =
																																				}cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0.6 ,  dataset_frame_num="<<dataset_frame_num<<",   count="<<count<<flush;}

	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * sizeof(float), fp32_k2k, 		0, NULL, &writeEvt);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSE3(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0.7 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData(..)
	cl_int 				res;
	//size_t local_size = local_work_size;
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	    uint	    layer,		                    //0
    res = clSetKernelArg(se3_grad_kernel, 1, sizeof(cl_mem), &mipmap_buf);				                        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	                //1
	res = clSetKernelArg(se3_grad_kernel, 2, sizeof(cl_mem), &uint_param_buf);						            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(se3_grad_kernel, 3, sizeof(cl_mem), &fp32_param_buf);						            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//3
	res = clSetKernelArg(se3_grad_kernel, 4, sizeof(cl_mem), &k2kbuf);								            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//4		// TODO keyframe2K
	res = clSetKernelArg(se3_grad_kernel, 5, sizeof(cl_mem), &keyframe_imgmem);				            		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//5		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	res = clSetKernelArg(se3_grad_kernel, 6, sizeof(cl_mem), &imgmem);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem,							//6
	res = clSetKernelArg(se3_grad_kernel, 7, sizeof(cl_mem), &keyframe_SE3_grad_map_mem);	            		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		keyframe_SE3_grad_map_mem		//7
	res = clSetKernelArg(se3_grad_kernel, 8, sizeof(cl_mem), &SE3_grad_map_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_grad_map					//8
	res = clSetKernelArg(se3_grad_kernel, 9, sizeof(cl_mem), &keyframe_depth_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float*		keyframe_depth_mem				//9		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	res = clSetKernelArg(se3_grad_kernel,10, local_work_size*6*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_grads					//10	6 DoF, float4 channels
	res = clSetKernelArg(se3_grad_kernel,11, sizeof(cl_mem), &se3_sum_mem);		 					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_grads,				//11
	res = clSetKernelArg(se3_grad_kernel,12, sizeof(cl_mem), &SE3_incr_map_mem);					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_incr_map_					//12
	res = clSetKernelArg(se3_grad_kernel,13, sizeof(cl_mem), &SE3_rho_map_mem);					                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_					        //13

	res = clSetKernelArg(se3_grad_kernel,14, local_work_size*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_rho_sq				//14	1 DoF, float4 channels
	res = clSetKernelArg(se3_grad_kernel,15, sizeof(cl_mem), &se3_sum_rho_sq_mem);		 					    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_rho_sq,				//15

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk1 ."<<flush;}
	mipmap_call_kernel( se3_grad_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_iter_"<< count << "_estimateSE3_";
                                                                                                                                                stringstream ss_path;
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_incr_map_mem, ss.str(), paths.at("SE3_incr_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );

																																				cout<<"\n\nRunCL::estimateSE3(..)_chk3.6 ."<<flush;
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at("SE3_rho_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 1 );
                                                                                                                                                cout<<"\n\nRunCL::estimateSE3(..)_chk3.7 ."<<flush;

                                                                                                                                                //DownloadAndSave(	SE3_incr_map_mem, ss.str(), paths.at("gxmem[frame_bool_idx]"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false, 1 );
                                                                                                                                                /*
                                                                                                                                                 * DownloadAndSave_3Channel(	gxmem[frame_bool_idx], ss.str(), paths.at("gxmem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
                                                                                                                                                 * DownloadAndSave_3Channel(	gymem[frame_bool_idx], ss.str(), paths.at("gymem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
                                                                                                                                                 * DownloadAndSave_3Channel(	g1mem[frame_bool_idx], ss.str(), paths.at("g1mem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
                                                                                                                                                 */
																																			}//(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range )
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk4 ."<<flush;}
	//	mipmap_params set in mipmap_call_kernel(..) below																					                                                          __constant 	uint*		mipmap_params,	//0
	res = clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), &uint_param_buf);                                    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant 	uint*		uint_params,	//1
	res = clSetKernelArg(reduce_kernel, 2, sizeof(cl_mem), &se3_sum_mem);                                       if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float8*		se3_sum			//2
	res = clSetKernelArg(reduce_kernel, 3, local_work_size*4*sizeof(float), 	NULL);	                                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local    	float8*		local_sum_grads	//3
	res = clSetKernelArg(reduce_kernel, 4, sizeof(cl_mem), &se3_sum2_mem);                                      if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float8*		se3_sum2,		//4

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk5 ."<<flush;}
                                                                                                                                            // directly read higher layers
	uint num_DoFs = 6;
    cv::Mat se3_sum_mat = cv::Mat::zeros (se3_sum_size, num_DoFs*4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)				// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( se3_sum_mat.data, se3_sum_mem, se3_sum_size_bytes );                                                                        // se3_sum_size_bytes
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nse3_sum_mat.size()="<<se3_sum_mat.size()<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																			}
                                                                                                                                            if(verbosity>local_verbosity_threshold+1) {cout<<"\n\nRunCL::estimateSE3(..)_chk5 ."<<flush;
                                                                                                                                                cout << "\n\n se3_sum_mat.data = (\n";
                                                                                                                                                for (int i=0; i<se3_sum_size; i++){
                                                                                                                                                    cout << "\n group="<<i<<" : ( " << flush;
                                                                                                                                                    for (int j=0; j<8; j++){
                                                                                                                                                                                                    //cout << ",  \nse3_sum_mat.at<float>("<<i<<","<<j<<") = " << flush;
                                                                                                                                                        cout << se3_sum_mat.at<float>(i,j) << " , " << flush;
                                                                                                                                                    }
                                                                                                                                                    cout << ")" << flush;
                                                                                                                                                }cout << "\n)\n" << flush;
                                                                                                                                                cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
                                                                                                                                            }                                                       // if start, stop  larger layers, call reduce kernel. ? cut off between large vs small layers ?
    //float SE3_results[8][6][4] = {{{0}}}; 																									// max 8 layers, 6+1 DoF, 4 channels
																																			if(verbosity>local_verbosity_threshold+1) {cout<<"\n\nRunCL::estimateSE3(..)_chk6 ."<<flush;
																																				cout << "\n\nse3_sum_mat.at<float> (i*num_DoFs + j,  k) ";
																																				for (int i=0; i< se3_sum_size ; i++){
																																					cout << "\ni="<<i<<":   ";
																																					for (int j=0; j<num_DoFs; j++){
																																						cout << ",     (";
																																						for (int k=0; k<4; k++){
																																							cout << "," << se3_sum_mat.at<float> (i , j*4 + k)  ;
																																						}cout << ")";
																																					}cout << flush;
																																				}
																																				cout << endl << endl;
																																			}

	for (int i=0; i<=mm_num_reductions+1; i++){
        uint read_offset_ 		= MipMap[i*8 + MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
        uint global_sum_offset 	= read_offset_ / local_work_size ;
        uint groups_to_sum 		= se3_sum_mat.at<float>(global_sum_offset, 0);
        uint start_group 		= global_sum_offset + 1;
        uint stop_group 		= start_group + groups_to_sum ;   // -1																		// skip the last group due to odd 7th value.
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\ni="<<i<<", read_offset_="<<read_offset_<<",  global_sum_offset="<<global_sum_offset<<",  groups_to_sum="<<groups_to_sum<< ",  start_group="<<start_group<<",  stop_group="<<stop_group;
																																			}
		for (int j=start_group; j< stop_group  ; j++){
			for (int k=0; k<num_DoFs; k++){
				for (int l=0; l<4; l++){
					SE3_results[i][k][l] += se3_sum_mat.at<float>(j, k*4 + l); 		//l =4 =num channels														// sum j groups for this layer of the MipMap.			// se3_sum_mat.at<float>(j, k);
				}
			}
		}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nLayer "<<i<<" SE3_results = (";																// raw results
																																				for (int k=0; k<num_DoFs; k++){
																																					cout << "(";
																																					for (int l=0; l<4; l++){
																																						cout << ", " << SE3_results[i][k][l] ;
																																					}cout << ")";
																																				}cout << ")";
																																			}
    }
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << endl;
																																				for (int i=0; i<=mm_num_reductions+1; i++){ 															// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<i<<" SE3_results/num_groups = (";
																																					for (int k=0; k<num_DoFs; k++){
																																						cout << "(";
																																						for (int l=0; l<3; l++){
																																							cout << ", " << SE3_results[i][k][l] / ( SE3_results[i][k][3]  *  img_stats[IMG_VAR+l]  );	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}
																																						cout << ", " << SE3_results[i][k][3] << ")";
																																					}cout << ")";
																																				}
																																			}

    cv::Mat rho_sq_sum_mat = cv::Mat::zeros (se3_sum_size, 4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)					// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( rho_sq_sum_mat.data, se3_sum_rho_sq_mem, pix_sum_size_bytes );
	//float Rho_sq_reults[8][4] = {{0}};
	for (int i=0; i<=mm_num_reductions+1; i++){
		uint read_offset_ 			= MipMap[i*8 +MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
		uint global_sum_offset 		= read_offset_ / local_work_size ;
		uint groups_to_sum 			= se3_sum_mat.at<float>(global_sum_offset, 0);
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
