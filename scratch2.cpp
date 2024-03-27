// old functions from fp32 branch

void Dynamic_slam::estimateSE3(){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::estimateSE3()_chk 0" << flush;}
																																			// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
	float Rho_sq_result=FLT_MAX,   old_Rho_sq_result=FLT_MAX,   next_layer_Rho_sq_result=FLT_MAX;
	uint layer = 5;
	float factor = 0.005;
	for (int iter = 0; iter<10; iter++){ 																									// TODO step down layers if fits well enough, and out if fits before iteration limit. Set iteration limit param in config.json file.
		if (iter%2==0 && layer>1) {
			layer--;
			old_Rho_sq_result = next_layer_Rho_sq_result;
		}																																	if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3()_chk 0.5: iter="<<iter<<",  layer="<<layer << flush;}
		float SE3_reults[8][6][4] = {{{0}}};
		float Rho_sq_results[8][4] = {{0}};
		runcl.estimateSE3(SE3_reults, Rho_sq_results, iter, 0, 8);																			//RunCL::estimateSE3(uint start=0, uint stop=8);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << endl;
																																				for (int i=0; i<=runcl.mm_num_reductions+1; i++){ 															// results / (num_valid_px * img_variance)
																																					cout << "\nDynamic_slam::estimateSE3(), Layer "<<i<<" SE3_results/num_groups = (";
																																					for (int k=0; k<6; k++){
																																						cout << "(";
																																						for (int l=0; l<3; l++){
																																							cout << ", " << SE3_reults[i][k][l] / ( SE3_reults[i][k][3]  *  runcl.img_stats[IMG_VAR+l]  );	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}
																																						cout << ", " << SE3_reults[i][k][3] << ")";
																																					}cout << ")";
																																				}
																																			}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << endl;
																																				for (int i=0; i<=runcl.mm_num_reductions+1; i++){ 															// results / (num_valid_px * img_variance)
																																					cout << "\nDynamic_slam::estimateSE3(), Layer "<<i<<" mm_num_reductions = "<< runcl.mm_num_reductions <<",  Rho_sq_results/num_groups = (";
																																					if (Rho_sq_results[i][3] > 0){
																																						for (int l=0; l<3; l++){  cout << ", " << Rho_sq_results[i][l] / ( Rho_sq_results[i][3]  *  runcl.img_stats[IMG_VAR+l]  );	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}cout << ", " << Rho_sq_results[i][3] << ")";
																																					}else{
																																						for (int l=0; l<3; l++){  cout << ", " << 0.0f  ;	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}cout << ", " << Rho_sq_results[i][3] << ")";
																																					}
																																				}
																																			}
		uint channel  = 0; 																													// TODO combine Rho HSV channels
		Rho_sq_result = Rho_sq_results[layer][channel] / ( Rho_sq_results[layer][3]  *  runcl.img_stats[IMG_VAR+channel] );
		if (layer >0) { next_layer_Rho_sq_result  = Rho_sq_results[layer+1][channel] / ( Rho_sq_results[layer+1][3]  *  runcl.img_stats[IMG_VAR+channel] );}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\niter="<<iter<<", layer="<<layer<<", old_Rho_sq_result="<<old_Rho_sq_result<<",  Rho_sq_result="<<Rho_sq_result <<",  next_layer_Rho_sq_result="<< next_layer_Rho_sq_result <<flush;
																																			}
		if (iter>0 && Rho_sq_result > old_Rho_sq_result) {																					// If new sample is worse, reject it. Continue to next iter. ? try a smaller step e.g. half size ?
																																			if(verbosity>local_verbosity_threshold) {cout << " (iter>0 && Rho_sq_result > old_Rho_sq_result)" << flush;}
			//continue;
		}
		old_Rho_sq_result = Rho_sq_result;

		float SE3_incr[6];
		//uint channel = 0;
		for (int SE3=0; SE3<6; SE3++) {SE3_incr[SE3] = SE3_reults[5][SE3][channel] / ( SE3_reults[5][SE3][3]  *  runcl.img_stats[IMG_VAR+channel]  );}																// For initial example take layer , channel[0] for each SE3 DoF.
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3()_chk 1" << flush;
																																				cout << "\n\nSE3_incr[SE3] = "; 	for (int SE3=0; SE3< 6; SE3++) cout << ", " << SE3_incr[SE3];
																																				cout << "\n\nold pose2pose = "; 	for (int SE3=0; SE3<16; SE3++) cout << ", " << pose2pose.operator()(SE3/4,SE3%4);
																																				cout << "\n\nold K2K = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", " << K2K.operator()(SE3/4,SE3%4);
																																			}
		if (layer==1) factor *= 0.65;
		Matx61f update;
		for (int SE3=0; SE3<6; SE3++) {update.operator()(SE3) = factor * SE3_reults[layer][SE3][channel] / ( SE3_reults[layer][SE3][3] * runcl.img_stats[IMG_VAR+channel] ); }
		cv::Matx44f SE3Incr_matx = SE3_Matx44f(update);

		pose2pose = pose2pose *  SE3Incr_matx;
		K2K = old_K * pose2pose * inv_K;
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3()_chk 2" << flush;
																																				cout <<"\nupdate = ";
																																				for (int i=0; i<6; i++){cout << ", " << update.operator()(0,i);}
																																				cout << flush;

																																				cout << "\nSE3Incr_matx = ";
																																				for (int i=0; i<4; i++){
																																					cout << "(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< SE3Incr_matx.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\nNew K2K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< K2K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\nNew pose2pose = ";
																																				for (int i=0; i<4; i++){
																																					cout << "(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< pose2pose.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\nold_K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< old_K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\ninv_K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< inv_K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																			}
		for (int i=0; i<16; i++){ runcl.fp32_k2k[i] = K2K.operator()(i/4, i%4);   															}//if(verbosity>local_verbosity_threshold) cout << "\n\nK2K ("<<i%4 <<","<< i/4<<") = "<< runcl.fp32_k2k[i]; }
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3()_chk 3\n" << flush;
																																				cout << "\n\nupdate.operator()(SE3) = "; 	for (int SE3=0; SE3< 6; SE3++) cout << ", " << update.operator()(SE3);
																																				cout << "\n\nSE3Incr_matx = "; 				for (int SE3=0; SE3<16; SE3++) cout << ", " << SE3Incr_matx.operator()(SE3/4,SE3%4);
																																				cout << "\n\nnew pose2pose = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", " << pose2pose.operator()(SE3/4,SE3%4);
																																				cout << "\n\nnew K2K = "; 					for (int SE3=0; SE3<16; SE3++) cout << ", " << K2K.operator()(SE3/4,SE3%4);
																																			}
		// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
		// # Pass prediction to lower layers. Does it fit better ?
		// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
	}
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3()_chk 3\n" << flush;
																																				cout << "\nruncl.frame_num = "<<runcl.frame_num;
																																				cout << "\npose2pose_accumulated = ";
																																				for (int i=0; i<4; i++){
																																					cout << "(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< pose2pose_accumulated.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\nNew pose2pose = ";
																																				for (int i=0; i<4; i++){
																																					cout << "(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< pose2pose.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																			}
	if (runcl.frame_num > 0 ) pose2pose_accumulated = pose2pose_accumulated * pose2pose;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::estimateSE3()_chk 4  Finished\n" << flush;}
}


//#########################################
void RunCL::estimateSE3(float SE3_results[8][6][4], float Rho_sq_results[8][4], int count, uint start, uint stop){ //estimateSE3(); 	(uint start=0, uint stop=8)			// TODO replace arbitrary fixed constant with a const uint variable in the header...
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0 ."<<flush;}
    cl_event writeEvt;
    cl_int status;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::estimateSE3(..)__chk_0.5: K2K= ";
																																				for (int i=0; i<16; i++){
																																					cout << ",  "<< fp32_k2k[i]; // K2K ("<<i%4 <<","<< i/4<<") =
																																				}cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0.6 ,  frame_num="<<frame_num<<",   count="<<count<<flush;}

	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * sizeof(float), fp32_k2k, 		0, NULL, &writeEvt);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSE3(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0.7 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData(..)
	cl_int 				res;
	//size_t local_size = local_work_size;
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	    uint	    layer,		                    //0
    res = clSetKernelArg(se3_grad_kernel, 1, sizeof(cl_mem), &mipmap_buf);				                        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	                //1
	res = clSetKernelArg(se3_grad_kernel, 2, sizeof(cl_mem), &uint_param_buf);						            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(se3_grad_kernel, 3, sizeof(cl_mem), &fp32_param_buf);						            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//3
	res = clSetKernelArg(se3_grad_kernel, 4, sizeof(cl_mem), &k2kbuf);								            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//4
	res = clSetKernelArg(se3_grad_kernel, 5, sizeof(cl_mem), &imgmem[frame_bool_idx]);				            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem[frame_bool_idx],			//5
	res = clSetKernelArg(se3_grad_kernel, 6, sizeof(cl_mem), &imgmem[!frame_bool_idx]);				            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem[!frame_bool_idx],		//6
	res = clSetKernelArg(se3_grad_kernel, 7, sizeof(cl_mem), &SE3_grad_map_mem[frame_bool_idx]);	            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_grad_map[frame_bool_idx]	//7
	res = clSetKernelArg(se3_grad_kernel, 8, sizeof(cl_mem), &SE3_grad_map_mem[!frame_bool_idx]);	            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_grad_map[!frame_bool_idx]	//8
	res = clSetKernelArg(se3_grad_kernel, 9, sizeof(cl_mem), &depth_mem);						                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		depth_map					    //9		// NB GT_depth, not inv_depth depth_mem
	res = clSetKernelArg(se3_grad_kernel,10, local_work_size*7*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_grads					//10	6 DoF, float4 channels
	res = clSetKernelArg(se3_grad_kernel,11, sizeof(cl_mem), &se3_sum_mem);		 					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_grads,				//11
	res = clSetKernelArg(se3_grad_kernel,12, sizeof(cl_mem), &SE3_incr_map_mem);					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_incr_map_					//12
	res = clSetKernelArg(se3_grad_kernel,13, sizeof(cl_mem), &SE3_rho_map_mem);					                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_					        //13

	res = clSetKernelArg(se3_grad_kernel,14, local_work_size*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_rho_sq				//14	1 DoF, float4 channels
	res = clSetKernelArg(se3_grad_kernel,15, sizeof(cl_mem), &se3_sum_rho_sq_mem);		 					    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_rho_sq,				//15


																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk1 ."<<flush;}
	mipmap_call_kernel( se3_grad_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << frame_num << "_iter_"<< count << "_img_grad";
                                                                                                                                                stringstream ss_path;
                                                                                                                                                ss_path << "SE3_incr_map_mem";
                                                                                                                                                cout << "\n" << ss_path.str() <<flush;
                                                                                                                                                cout << "\n" <<  paths.at(ss_path.str()) <<flush;
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_incr_map_mem, ss.str(), paths.at(ss_path.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );

                                                                                                                                                cout<<"\n\nRunCL::estimateSE3(..)_chk3.5 ."<<flush;
                                                                                                                                                stringstream ss_path_rho;
                                                                                                                                                ss_path_rho << "SE3_rho_map_mem";
                                                                                                                                                cout << "\n" << ss_path_rho.str() <<flush;
                                                                                                                                                cout << "\n" <<  paths.at(ss_path_rho.str()) <<flush;

																																				cout<<"\n\nRunCL::estimateSE3(..)_chk3.6 ."<<flush;
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at(ss_path_rho.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 1 );
                                                                                                                                                cout<<"\n\nRunCL::estimateSE3(..)_chk3.7 ."<<flush;

                                                                                                                                                //DownloadAndSave(	SE3_incr_map_mem, ss.str(), paths.at("gxmem[frame_bool_idx]"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false, 1 );
                                                                                                                                                /*
                                                                                                                                                 * DownloadAndSave_3Channel(	gxmem[frame_bool_idx], ss.str(), paths.at("gxmem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
                                                                                                                                                 * DownloadAndSave_3Channel(	gymem[frame_bool_idx], ss.str(), paths.at("gymem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
                                                                                                                                                 * DownloadAndSave_3Channel(	g1mem[frame_bool_idx], ss.str(), paths.at("g1mem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
                                                                                                                                                 */
																																			}//(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range )
                                                                                                                                            if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::estimateSE3(..)_chk4 ."<<flush;}
	//	mipmap_params set in mipmap_call_kernel(..) below																					                                                          __constant 	uint*		mipmap_params,	//0
	res = clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), &uint_param_buf);                                    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant 	uint*		uint_params,	//1
	res = clSetKernelArg(reduce_kernel, 2, sizeof(cl_mem), &se3_sum_mem);                                       if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float8*		se3_sum			//2
	res = clSetKernelArg(reduce_kernel, 3, local_work_size*8*sizeof(float), 	NULL);	                                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local    	float8*		local_sum_grads	//3
	res = clSetKernelArg(reduce_kernel, 4, sizeof(cl_mem), &se3_sum2_mem);                                      if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float8*		se3_sum2,		//4

																																			if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::estimateSE3(..)_chk5 ."<<flush;}
                                                                                                                                            // directly read higher layers
	uint num_DoFs = 6;
    cv::Mat se3_sum_mat = cv::Mat::zeros (se3_sum_size, num_DoFs*4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)				// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( se3_sum_mat.data, se3_sum_mem, se3_sum_size_bytes );                                                                        // se3_sum_size_bytes
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\nse3_sum_mat.size()="<<se3_sum_mat.size()<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																			}
                                                                                                                                            if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::estimateSE3(..)_chk5 ."<<flush;
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
																																			if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::estimateSE3(..)_chk6 ."<<flush;
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
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\ni="<<i<<", read_offset_="<<read_offset_<<",  global_sum_offset="<<global_sum_offset<<",  groups_to_sum="<<groups_to_sum<< ",  start_group="<<start_group<<",  stop_group="<<stop_group;
																																			}
        for (int j=start_group; j< stop_group  ; j++){
            for (int k=0; k<num_DoFs; k++){
				for (int l=0; l<4; l++){
					SE3_results[i][k][l] += se3_sum_mat.at<float>(j, k*4 + l); // se3_sum_mat.at<float>(j, k);                         		// sum j groups for this layer of the MipMap.
				}
            }
        }
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\nLayer "<<i<<" SE3_results = (";																// raw results
																																				for (int k=0; k<num_DoFs; k++){
																																					cout << "(";
																																					for (int l=0; l<4; l++){
																																						cout << ", " << SE3_results[i][k][l] ;
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
//##############################
uint global_id_u;
__kernel void se3_grad(
	__private	uint	layer,					//0
	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3
	__global	float16*k2k,					//4
	__global 	float4*	img_cur,				//5
	__global 	float4*	img_new,				//6
	__global 	float8*	SE3_grad_map_cur_frame,	//7
	__global 	float8*	SE3_grad_map_new_frame,	//8
	__global	float* 	depth_map,				//9		// NB GT_depth, not inv_depth
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

	uint depth_index	= v * reduction * base_cols + u * reduction;								// Sparse sampling of the depth map of the base image.
	float inv_depth 	= 1/depth_map[depth_index]; 												//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.
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
	/*
	if (global_id_u==1){
		printf("\nkernel se3_grad(..)  k2k_pvt=(%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f)"\
		,k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],   k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],   k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15]   );
	}
	*/
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
