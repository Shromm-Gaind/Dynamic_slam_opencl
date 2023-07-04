#include "RunCL.h"

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

using namespace std;

void RunCL::predictFrame(){ //predictFrame();


}

void RunCL::loadFrame(cv::Mat image){ //getFrame();
	int local_verbosity_threshold = 0;
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout << "\n RunCL::loadFrame_chk 0\n" << flush;}
	cl_int status;
	cl_event writeEvt;																										               // WriteBuffer basemem #########
	status = clEnqueueWriteBuffer(uload_queue, basemem, CL_FALSE, 0, image_size_bytes, image.data, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
                                                                                                                                            if (verbosity>local_verbosity_threshold){
                                                                                                                                                stringstream ss;	ss << frame_num << "loadFrame";
                                                                                                                                                DownloadAndSave_3Channel(basemem, ss.str(), paths.at("basemem"), image_size_bytes, baseImage_size,  baseImage_type, 	false );
                                                                                                                                            }
}

void RunCL::cvt_color_space(){ //getFrame(); basemem(CV_8UC3, RGB)->imgmem(CV16FC3, HSV), NB we will use basemem for image upload, and imgmem for the MipMap. RGB is default for .png standard.
	int local_verbosity_threshold = 0;
                                                                                                                                            if(verbosity>local_verbosity_threshold) {
                                                                                                                                                cout<<"\n\nRunCL::cvt_color_space()_chk0"<<flush;
                                                                                                                                                cout << "\n";
                                                                                                                                                cout << ",mm_Image_size = " << mm_Image_size << endl;
                                                                                                                                                cout << ",mm_Image_type = "	<< mm_Image_type << endl;
                                                                                                                                                cout << ",mm_size_bytes_C3 = " << mm_size_bytes_C3 << endl;
                                                                                                                                                cout << ",mm_size_bytes_C4 = " << mm_size_bytes_C4 << endl;
                                                                                                                                                cout << ",mm_size_bytes_C1 = " << mm_size_bytes_C1 << endl;
                                                                                                                                                cout << "\n";
                                                                                                                                                cout << ",baseImage_size, = " << baseImage_size << endl;
                                                                                                                                                cout << ",baseImage_type = " << baseImage_type << endl;
                                                                                                                                                cout << ",image_size_bytes = " << image_size_bytes	<< endl;
                                                                                                                                                cout << ",mm_vol_size_bytes = " << mm_vol_size_bytes << endl;
                                                                                                                                                cout << "\n" << flush;
                                                                                                                                            }
	//args
	cl_int res, status;
	cl_event ev;																																									// cvt_color_space_kernel  or  cvt_color_space_linear_kernel
	res = clSetKernelArg(cvt_color_space_linear_kernel, 0, sizeof(cl_mem), &basemem);					   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uchar3*		base,			//0
	res = clSetKernelArg(cvt_color_space_linear_kernel, 1, sizeof(cl_mem), &imgmem);	   				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global float4*		img,			//1	
	res = clSetKernelArg(cvt_color_space_linear_kernel, 2, sizeof(cl_mem), &uint_param_buf);			   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uint*		uint_params		//2
	res = clSetKernelArg(cvt_color_space_linear_kernel, 3, sizeof(cl_mem), &mipmap_buf);				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__constant uint*		mipmap_params,	//3 // NB layer = 0.
	res = clSetKernelArg(cvt_color_space_linear_kernel, 4, local_work_size*4*sizeof(float), 	NULL);	   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		local_sum_pix	//4
	res = clSetKernelArg(cvt_color_space_linear_kernel, 5, sizeof(cl_mem), &pix_sum_mem);				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		global_sum_pix	//5
	
	status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
																															if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::cvt_color_space()_chk1,  global_work_size="<< global_work_size <<flush;
	res = clEnqueueNDRangeKernel(m_queue, cvt_color_space_linear_kernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, &ev); 							// run cvt_color_space_kernel  aka cvt_color_space(..) ##### TODO which CommandQueue to use ? What events to check ?
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status = clFlush(m_queue);				                                                               if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents (1, &ev);		                                                               if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
                                                                                                                                            if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::cvt_color_space()_chk2"<<flush;
                                                                                                                                            if (verbosity>local_verbosity_threshold){
                                                                                                                                                stringstream ss;		ss << frame_num << "_cvt_color_space";
                                                                                                                                                stringstream ss_path;	ss_path << "imgmem";
                                                                                                                                                
                                                                                                                                                cv::Size new_Image_size = cv::Size(mm_width, mm_height);
                                                                                                                                                size_t   new_size_bytes = mm_width * mm_height * 4* 4;
                                                                                                                                                
                                                                                                                                                cout << "imgmem="<< imgmem << endl << flush;
                                                                                                                                                cout <<", ss.str()="<< ss.str() << endl << flush;
                                                                                                                                                cout <<", paths.at(\"imgmem\")="<< paths.at("imgmem") << endl << flush;
                                                                                                                                                
                                                                                                                                                cout <<", paths.at(" << ss_path.str() <<")="<< paths.at(ss_path.str()) << endl << flush;
                                                                                                                                                cout <<", new_size_bytes="<< new_size_bytes << endl << flush;
                                                                                                                                                cout <<", new_Image_size="<< new_Image_size <<"" << endl << flush;
                                                                                                                                                
                                                                                                                                                DownloadAndSave_3Channel(	imgmem, ss.str(), paths.at( ss_path.str() ), new_size_bytes/*mm_size_bytes_C4*/, new_Image_size/*mm_Image_size*/,  CV_32FC4 /*mm_Image_type*/, 	false );
                                                                                                                                                /*
                                                                                                                                                cout<<"\n\n,chk2.3,"<<flush;
                                                                                                                                                cout<<"\n img_sum_buf="<< img_sum_buf <<flush;
                                                                                                                                                cout<<"\n ss.str()="<< ss.str() <<flush;
                                                                                                                                                cout<<"\n paths.at(\"img_sum_buf\")="<< paths.at("img_sum_buf") <<flush;
                                                                                                                                                cout<<"\n mm_size_bytes_C1="<< mm_size_bytes_C1 <<flush;
                                                                                                                                                cout<<"\n mm_size_bytes_C3="<< mm_size_bytes_C3 <<flush;
                                                                                                                                                cout<<"\n mm_Image_size="<< mm_Image_size <<"\n\n"<<flush;
                                                                                                                                                */
                                                                                                                                                //DownloadAndSave_3Channel(	img_sum_buf, ss.str(), paths.at("img_sum_buf"),  mm_size_bytes_C3*2, mm_Image_size,  CV_32FC3 /*mm_Image_type*/, 	false ); // only when debugging.
                                                                                                                                            }
	cv::Mat pix_sum_mat = cv::Mat::zeros (pix_sum_size, 1, CV_32FC4); // cv::Mat::zeros (int rows, int cols, int type)						// NB the data returned is one float4 per group, for the base image, holding hsv channels plus entry[3]=pixel count.
	ReadOutput( pix_sum_mat.data, pix_sum_mem, pix_sum_size_bytes );                                                                        // se3_sum_size_bytes
	
                                                                                                                                            if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::cvt_color_space(..)_chk2 ."<<flush;
																																				cout << "\npix_sum_mat.size()="<<pix_sum_mat.size()<<flush;
																																				cout << "\npix_sum_size="<<pix_sum_size<<flush;
                                                                                                                                                cout << "\n pix_sum_mat.data = (\n";
                                                                                                                                                for (int i=0; i<pix_sum_size; i++){
                                                                                                                                                    cout << "\n group="<<i<<" : ( " << flush;
                                                                                                                                                    for (int j=0; j<4; j++){
                                                                                                                                                        cout << pix_sum_mat.at<float>(i,j) << " , " << flush;
                                                                                                                                                    }
                                                                                                                                                    cout << ")" << flush;
                                                                                                                                                }cout << "\n)\n" << flush;
                                                                                                                                            } 
	float pix_sum_reults[4] = {0};
	uint groups_to_sum = pix_sum_mat.at<float>(0, 0);
	uint start_group   = 1;
	uint stop_group    = start_group + groups_to_sum;
																																			if(verbosity>local_verbosity_threshold+2) cout << "\ngroups_to_sum="<<groups_to_sum<<",  stop_group="<<stop_group<<endl<<flush;
        
	for (int j=start_group; j< stop_group  ; j++){
		for (int k=0; k<4; k++){
			pix_sum_reults[k] += pix_sum_mat.at<float>(j, k); 
		}
	}
	uint layer =0;
	for (int i=0; i<3; i++){
		img_stats[layer*4 + IMG_MEAN + i ]	=	pix_sum_reults[i] / pix_sum_reults[3];
	}
	
	cl_event writeEvt;																										               // Upload img_mean to GPU
	status = clEnqueueWriteBuffer(uload_queue, img_stats_buf, CL_FALSE, 0, img_stats_size_bytes, img_stats, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
																																			if(verbosity>local_verbosity_threshold+2){
																																				cout << "\n Pix_sum_results = (";
																																				for (int k=0; k<4; k++){
																																						cout << ", " << pix_sum_reults[k] ;
																																				}cout << ")";
																																				cout << endl;
																																				cout << "\n Pix_sum_results/num_groups = (";
																																				for (int k=0; k<4; k++){
																																					cout << ", " << pix_sum_reults[k]/pix_sum_reults[3] ;
																																				}cout << ")";
																																			}
																																		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::cvt_color_space()_chk3_Finished"<<flush;
	// TODO NB it would be faster to find the mean from the smallest layer, BUT only if there are no bugs e.g. the black bottom edge.
	// Variance however must be computed for each layer, because blurring may reduce contrast &=> variance.
}

void RunCL::img_variance(){
	int local_verbosity_threshold = 1;
                          
	// TODO ? create a class for data, holding buffer, CPU data, stats about the data object, functions for write, read, save, display, & set_kernel_arg ? 
	
	cl_int res, status;
	cl_event ev, writeEvt;																																								// cvt_color_space_kernel  or  img_variance_kernel
	res = clSetKernelArg(img_variance_kernel, 0, sizeof(cl_mem), &img_stats_buf);						   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uchar3*		img_stats,		//0
	res = clSetKernelArg(img_variance_kernel, 1, sizeof(cl_mem), &imgmem);  			   				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global float4*		img,			//1	
	res = clSetKernelArg(img_variance_kernel, 2, sizeof(cl_mem), &uint_param_buf);						   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uint*		uint_params		//2
	res = clSetKernelArg(img_variance_kernel, 3, sizeof(cl_mem), &mipmap_buf);							   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__constant uint*		mipmap_params,	//3 // NB layer = 0.
	res = clSetKernelArg(img_variance_kernel, 4, local_work_size*4*sizeof(float), 	NULL);				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		local_sum_pix	//4
	res = clSetKernelArg(img_variance_kernel, 5, sizeof(cl_mem), &var_sum_mem);							   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		global_sum_pix	//5
	
	status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk1,  global_work_size="<< global_work_size <<flush;
	res = clEnqueueNDRangeKernel(m_queue, img_variance_kernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, &ev); 					// run img_variance _kernel  aka img_variance(..) ##### TODO which CommandQueue to use ? What events to check ?
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status = clFlush(m_queue);				                                                               if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents (1, &ev);		                                                               if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
                                                                                                                                            if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk2"<<flush;
	cv::Mat var_sum_mat = cv::Mat::zeros (pix_sum_size, 1, CV_32FC4); // cv::Mat::zeros (int rows, int cols, int type)						// NB the data returned is one float4 per group, for the base image, holding hsv channels plus entry[3]=pixel count.
	ReadOutput( var_sum_mat.data, var_sum_mem, pix_sum_size_bytes );                                                                        // se3_sum_size_bytes
                                                                                                                                            if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::img_variance(..)_chk2 ."<<flush;
																																				cout << "\nvar_sum_mat.size()="<<var_sum_mat.size()<<flush;
																																				cout << "\npix_sum_size="<<pix_sum_size<<flush;
                                                                                                                                                cout << "\n var_sum_mat.data = (\n";
                                                                                                                                                for (int i=0; i<pix_sum_size; i++){
                                                                                                                                                    cout << "\n group="<<i<<" : ( " << flush;
                                                                                                                                                    for (int j=0; j<4; j++){
                                                                                                                                                        cout << var_sum_mat.at<float>(i,j) << " , " << flush;
                                                                                                                                                    }
                                                                                                                                                    cout << ")" << flush;
                                                                                                                                                }cout << "\n)\n" << flush;
                                                                                                                                            } 
	float var_sum_reults[4] = {0};
	uint groups_to_sum = var_sum_mat.at<float>(0, 0);
	uint start_group   = 1;
	uint stop_group    = start_group + groups_to_sum;
																																			if(verbosity>local_verbosity_threshold+2) cout << "\ngroups_to_sum="<<groups_to_sum<<",  stop_group="<<stop_group<<endl<<flush;
	for (int j=start_group; j< stop_group  ; j++){
		for (int k=0; k<4; k++){
			var_sum_reults[k] += var_sum_mat.at<float>(j, k); 
		}
	}
	uint layer =0;
	for (int i=0; i<3; i++){
		img_stats[layer*4 + IMG_VAR + i ]	=	var_sum_reults[i] / var_sum_reults[3];
	}
																																			// Upload img_variance to GPU 
	status = clEnqueueWriteBuffer(uload_queue, img_stats_buf, CL_FALSE, 0, img_stats_size_bytes, img_stats, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n Var_sum_results = (";
																																				for (int k=0; k<4; k++){
																																						cout << ", " << var_sum_reults[k] ;
																																				}cout << ")";
																																				cout << endl;
																																				cout << "\n Var_sum_results/num_groups = (";
																																				for (int k=0; k<4; k++){
																																					cout << ", " << var_sum_reults[k]/var_sum_reults[3] ;
																																				}cout << ")";
																																			}
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk3_Finished"<<flush;
}

void RunCL::convert_depth(uint invert, float factor){
	int local_verbosity_threshold = 0;																										if(verbosity>local_verbosity_threshold) {
																																					cout<<"\n\nRunCL::convert_depth(uint invert, float factor)_chk0"<<flush;
																																					cout<<", invert="<<invert<<",  factor="<<factor<<flush;
																																			}
	cl_event 	ev;
	cl_int 		res, status;
	
	res = clSetKernelArg(convert_depth_kernel, 0, sizeof(uint),   &invert );			if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__private	 bool 	invert,			//0
	res = clSetKernelArg(convert_depth_kernel, 1, sizeof(float),  &factor );			if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__private	 float 	factor,			//1
	res = clSetKernelArg(convert_depth_kernel, 2, sizeof(cl_mem), &mipmap_buf);			if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__constant uint*	mipmap_params,	//2
	res = clSetKernelArg(convert_depth_kernel, 3, sizeof(cl_mem), &uint_param_buf);		if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__constant uint*	uint_params		//3
	res = clSetKernelArg(convert_depth_kernel, 4, sizeof(cl_mem), &depth_mem);			if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__global	 float* depth_map,		//4
	res = clSetKernelArg(convert_depth_kernel, 5, sizeof(cl_mem), &depth_mem_GT);		if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__global	 float* depth_map,		//5
	
	status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::convert_depth()_chk1,  global_work_size="<< global_work_size <<flush;
	
	res 	= clEnqueueNDRangeKernel(m_queue, convert_depth_kernel, 1, 0,  &global_work_size, &local_work_size, 0, NULL, &ev); // run mipmap_float4_kernel, NB wait for own previous iteration.
																						if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(m_queue);															if (status != CL_SUCCESS)	{ cout << "\nclFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status 	= clWaitForEvents (1, &ev);													if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
}

void RunCL::mipmap_linear(){
	int local_verbosity_threshold = 0;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap(..)_chk0"<<flush;}
	cl_event 	writeEvt;//, ev;
	cl_int 		res, status;
	//uint 		mipmap[8];
	float		a = float(0.0625);
	float		b = float(0.125);
	float		c = float(0.25);
	float 		gaussian[9] = {a, b, a, b, c, b, a , b, a };																																//  TODO load gaussian kernel & size from conf.json .
	if (mm_gaussian_size!=3) {cout<<"Error: (mm_gaussian_size!=3). Need to add code to malloc gaussian array. Probably with jsoncpp from 'conf.json' file." <<flush; exit(0); }
	
	status = clEnqueueWriteBuffer(uload_queue, gaussian_buf, CL_FALSE, 0, mm_gaussian_size*mm_gaussian_size*sizeof(float), gaussian, 0, NULL, &writeEvt);											// write mipmap_buf
	if (status != CL_SUCCESS){cout<<"\nstatus = "<<checkerror(status)<<"\n"<<flush; cout << "Error: RunCL::mipmap, clEnqueueWriteBuffer, mipmap_buf \n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	size_t local_size = local_work_size;																																							// set kernel args
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                                      __private	 uint	    layer,		    //0
    res = clSetKernelArg(mipmap_float4_kernel, 1, sizeof(cl_mem), 					 	&mipmap_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		mipmap_params,	//1
	res = clSetKernelArg(mipmap_float4_kernel, 2, sizeof(cl_mem), 					 	&gaussian_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant float*		gaussian,		//2
	res = clSetKernelArg(mipmap_float4_kernel, 3, sizeof(cl_mem), 					 	&uint_param_buf);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		uint_params,	//3
	res = clSetKernelArg(mipmap_float4_kernel, 4, sizeof(cl_mem), 						&imgmem);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global   float4*	img,			//4	
	res = clSetKernelArg(mipmap_float4_kernel, 5, (local_size+4) *5*4* sizeof(float), 	NULL);						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__local    float4*	local_img_patch //5
	
	mipmap_call_kernel( mipmap_float4_kernel, m_queue );// TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.
	
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << frame_num << "_mipmap_linear";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				stringstream ss_path;	ss_path << "imgmem";
																																				DownloadAndSave_3Channel( imgmem, ss.str(), paths.at(ss_path.str()), new_size_bytes, new_Image_size, CV_32FC4, false );
																																				cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap(..)_chk4 Finished"<<flush;}
}

void RunCL::mipmap_depthmap(cl_mem depthmap_){
	int local_verbosity_threshold = 0;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_depthmap(..)_chk0"<<flush;}
	cl_event 	writeEvt;//, ev;
	cl_int 		res, status;
	//uint 		mipmap[8];
	float		a = float(0.0625);
	float		b = float(0.125);
	float		c = float(0.25);
	float 		gaussian[9] = {a, b, a, b, c, b, a , b, a };																																//  TODO load gaussian kernel & size from conf.json .
	if (mm_gaussian_size!=3) {cout<<"Error:RunCL::mipmap_depthmap(..) (mm_gaussian_size!=3). Need to add code to malloc gaussian array. Probably with jsoncpp from 'conf.json' file." <<flush; exit(0); }
	
	status = clEnqueueWriteBuffer(uload_queue, gaussian_buf, CL_FALSE, 0, mm_gaussian_size*mm_gaussian_size*sizeof(float), gaussian, 0, NULL, &writeEvt);											// write mipmap_buf
	if (status != CL_SUCCESS){cout<<"\nstatus = "<<checkerror(status)<<"\n"<<flush; cout << "Error: RunCL::mipmap_depthmap, clEnqueueWriteBuffer, mipmap_buf \n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	size_t local_size = local_work_size;																																							// set kernel args
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                                      __private	 uint	    layer,		    //0
    res = clSetKernelArg(mipmap_float_kernel, 1, sizeof(cl_mem), 					 	&mipmap_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		mipmap_params,	//1
	res = clSetKernelArg(mipmap_float_kernel, 2, sizeof(cl_mem), 					 	&gaussian_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant float*		gaussian,		//2
	res = clSetKernelArg(mipmap_float_kernel, 3, sizeof(cl_mem), 					 	&uint_param_buf);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		uint_params,	//3
	res = clSetKernelArg(mipmap_float_kernel, 4, sizeof(cl_mem), 						&depthmap_);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global   float4*	img,			//4	
	res = clSetKernelArg(mipmap_float_kernel, 5, (local_size+4) *5*sizeof(float), 		NULL);						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__local    float4*	local_img_patch //5
	
	mipmap_call_kernel( mipmap_float_kernel, m_queue );// TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.
	
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap_depthmap(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << frame_num << "_mipmap_depthmap";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				stringstream ss_path;	ss_path << "imgmem";
																																				DownloadAndSave_3Channel( imgmem, ss.str(), paths.at(ss_path.str()), new_size_bytes, new_Image_size, CV_32FC4, false );
																																				cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_depthmap(..)_chk4 Finished"<<flush;}
}

void RunCL::img_gradients(){ //getFrame();
	int local_verbosity_threshold = 0;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk0"<<flush;}
	cl_int res;
	size_t num_threads = ceil( (float)(mm_layerstep)/(float)local_work_size ) * local_work_size ; 
																																			if(verbosity>local_verbosity_threshold) {cout << "\n num_threads = " << num_threads << ",   mm_layerstep = " << mm_layerstep << ",  local_work_size = " << local_work_size  <<endl << flush;}
	//res = clSetKernelArg(img_grad_kernel, 3, sizeof(cl_mem), &mipmap_buf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global uint*	mipmap_params,	//3
	
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	 uint	    layer,		//0
    res = clSetKernelArg(img_grad_kernel, 1, sizeof(cl_mem), &mipmap_buf);				                      if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*	mipmap_params,	//1
	res = clSetKernelArg(img_grad_kernel, 2, sizeof(cl_mem), &uint_param_buf);						          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*	uint_params		//2
	res = clSetKernelArg(img_grad_kernel, 3, sizeof(cl_mem), &fp32_param_buf);						          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant float*	fp32_params		//3
	res = clSetKernelArg(img_grad_kernel, 4, sizeof(cl_mem), &imgmem);				          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global   float4*	img,		//4
	res = clSetKernelArg(img_grad_kernel, 5, sizeof(cl_mem), &gxmem);				          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	gxp,		//5
	res = clSetKernelArg(img_grad_kernel, 6, sizeof(cl_mem), &gymem);				          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	gyp,		//6
	res = clSetKernelArg(img_grad_kernel, 7, sizeof(cl_mem), &g1mem);				          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	g1p			//7
	res = clSetKernelArg(img_grad_kernel, 8, sizeof(cl_mem), &SE3_map_mem);							          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant float2*	SE3_map,	//8
	//res = clSetKernelArg(img_grad_kernel, 9, sizeof(cl_mem), &depth_mem);							          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	float* 		depth_map,	//9		// NB GT_depth, not inv_depth
	res = clSetKernelArg(img_grad_kernel, 9, sizeof(cl_mem), &SE3_grad_map_mem);	          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	SE3_grad_map//9
	
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk2"<<flush;}
	mipmap_call_kernel( img_grad_kernel, m_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk3 Finished all loops. Saving gxmem, gymem, g1mem."<<flush;
																																				stringstream ss;	ss << frame_num << "_img_grad";
																																				stringstream ss_path;	
																																				ss_path << "gxmem"; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_3Channel(	gxmem, ss.str(), paths.at(ss_path.str()),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "gymem"; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_3Channel(	gymem, ss.str(), paths.at(ss_path.str()),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "g1mem"; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_3Channel(	g1mem, ss.str(), paths.at(ss_path.str()),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				///
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "SE3_grad_map_mem"<<flush; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_6Channel_volume(  SE3_grad_map_mem, ss.str(), paths.at(ss_path.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																				///
																																				cout << "\n\n SE3_grad_map_mem = SE3_grad_map_mem = "<<SE3_grad_map_mem;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk4 Finished."<<flush;}
}

void RunCL::load_GT_depth(cv::Mat GT_depth, bool invert){ //getFrameData();, cv::Matx44f GT_K2K,   cv::Matx44f GT_pose2pose
    int local_verbosity_threshold = 0;
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_0:"<<flush;
    //for (int i=0; i<16; i++){ fp32_k2k[i] = GT_K2K.operator()(i/4, i%4);   																if(verbosity>local_verbosity_threshold) cout << "\nRunCL::loadFrameData(..)_chk_1:  K2K ("<<i%4 <<","<< i/4<<") = "<< fp32_k2k[i]; }
    cl_event 		writeEvt;
	cl_int 	 		status;
	stringstream 	ss;
	ss << "__load_GT_depth" << (keyFrameCount*1000 + costVolCount);
	
    status = clEnqueueWriteBuffer(uload_queue, depth_mem, 		CL_FALSE, 0, image_size_bytes_C1,	 GT_depth.data, 0, NULL, &writeEvt);	
																									if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::load_GT_depth(..)_chk_2\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	ss << "__0";
																																		if(verbosity>local_verbosity_threshold+1)
																																			{ DownloadAndSave( depth_mem_GT,   	ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;}
	float factor = 1;//256;
	convert_depth( invert, factor);  
	ss << "__1";
																																		if(verbosity>local_verbosity_threshold+1)
																																			{ DownloadAndSave( depth_mem_GT,   	ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;}
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_1:"<<flush;
	mipmap_depthmap(depth_mem_GT);
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_2:"<<flush;
	ss << "__2";
																																		if(verbosity>local_verbosity_threshold)
																																			{ DownloadAndSave( depth_mem_GT,   	ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;}
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_finished:"<<flush;
}

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
																																																	stringstream ss;	ss << frame_num << "_SE3_map";
																																																	DownloadAndSave_2Channel_volume(SE3_map_mem, ss.str(), paths.at("SE3_map_mem"), mm_size_bytes_C1*2, mm_Image_size, CV_32FC2, false, 1.0, 6 /*SE3, 6DoF */);
																																																}
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precom_param_maps(float SE3_k2k[6*16])_chk.. Finished "<<flush;}
}

void RunCL::estimateSO3(float SO3_results[8][3][4], float Rho_sq_results[8][4], int count, uint start, uint stop){ //estimateSO3();	(uint start=0, uint stop=8)
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk0 ."<<flush;}
    cl_event writeEvt;
    cl_int status;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::estimateSO3(..)__chk_0.5: fp32_so3= ";
																																				for (int i=0; i<9; i++){ cout << ",  "<< fp32_so3_k2k[i]; }cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk0.6 ,  frame_num="<<frame_num<<",   count="<<count<<flush;}
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
	res = clSetKernelArg(so3_grad_kernel,9, local_work_size*3*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_grads					//9		3 DoF, float4 channels TODO why 7 ?
	res = clSetKernelArg(so3_grad_kernel,10, sizeof(cl_mem), &se3_sum_mem);		 					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_grads,				//10
	res = clSetKernelArg(so3_grad_kernel,11, sizeof(cl_mem), &SE3_incr_map_mem);					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_incr_map_					//11
	res = clSetKernelArg(so3_grad_kernel,12, sizeof(cl_mem), &SE3_rho_map_mem);					                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_					        //12
	res = clSetKernelArg(so3_grad_kernel,13, local_work_size*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_rho_sq				//13	1 DoF, float4 channels
	res = clSetKernelArg(so3_grad_kernel,14, sizeof(cl_mem), &se3_sum_rho_sq_mem);		 					    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_rho_sq,				//14
	
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk2 ."<<flush;}
	mipmap_call_kernel( so3_grad_kernel, m_queue, start, stop );
																																			
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSO3(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << frame_num << "_iter_"<< count << "_estimateSO3_";
																																				
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_incr_map_mem, ss.str(), paths.at("SO3_incr_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																			
																																				cout<<"\n\nRunCL::estimateSO3(..)_chk3.6 ."<<flush;
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at("SO3_rho_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 1 );
                                                                                                                                                
																																				//cout<<"\n\nRunCL::estimateSO3(..)_chk3.7 ."<<flush;
																																				//DownloadAndSave_3Channel_volume(  keyframe_imgmem,  ss.str(), paths.at("keyframe_imgmem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, 1, 1 );
																																				
																																				cout<<"\n\nRunCL::estimateSO3(..)_chk3.8 ."<<flush;
																																				DownloadAndSave_3Channel_volume(  imgmem,  ss.str(), paths.at("imgmem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, 1, 1 );
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
	res = clSetKernelArg(se3_grad_kernel, 4, sizeof(cl_mem), &k2kbuf);								            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//4		// TODO keyframe2K
	res = clSetKernelArg(se3_grad_kernel, 5, sizeof(cl_mem), &keyframe_imgmem);				            		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//5		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	res = clSetKernelArg(se3_grad_kernel, 6, sizeof(cl_mem), &imgmem);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem,							//6
	res = clSetKernelArg(se3_grad_kernel, 7, sizeof(cl_mem), &keyframe_SE3_grad_map_mem);	            		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		keyframe_SE3_grad_map_mem		//7
	res = clSetKernelArg(se3_grad_kernel, 8, sizeof(cl_mem), &SE3_grad_map_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_grad_map					//8
	res = clSetKernelArg(se3_grad_kernel, 9, sizeof(cl_mem), &keyframe_depth_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float*		keyframe_depth_mem				//9		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	res = clSetKernelArg(se3_grad_kernel,10, local_work_size*7*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_grads					//10	6 DoF, float4 channels
	res = clSetKernelArg(se3_grad_kernel,11, sizeof(cl_mem), &se3_sum_mem);		 					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_grads,				//11
	res = clSetKernelArg(se3_grad_kernel,12, sizeof(cl_mem), &SE3_incr_map_mem);					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_incr_map_					//12
	res = clSetKernelArg(se3_grad_kernel,13, sizeof(cl_mem), &SE3_rho_map_mem);					                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_					        //13
	
	res = clSetKernelArg(se3_grad_kernel,14, local_work_size*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_rho_sq				//14	1 DoF, float4 channels
	res = clSetKernelArg(se3_grad_kernel,15, sizeof(cl_mem), &se3_sum_rho_sq_mem);		 					    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_rho_sq,				//15
	
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk1 ."<<flush;}
	mipmap_call_kernel( se3_grad_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << frame_num << "_iter_"<< count << "_estimateSE3_";
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

void RunCL::tracking_result(string result){
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::tracking_result(..)_chk0"<<flush;
																																				stringstream ss;			ss << frame_num << "_iter_"<< count << "_img_grad_" << result;
																																				stringstream ss_path_rho;	ss_path_rho << "SE3_rho_map_mem";
																																				cout << " , " << ss_path_rho.str() << " , " <<  paths.at(ss_path_rho.str()) << " , " << ss.str()  <<flush;
																																				DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at(ss_path_rho.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 1 );
																																			}
}

void RunCL::estimateCalibration(){ //estimateCalibration(); 		// own thread, one iter.


}	

void RunCL::transform_depthmap( cv::Matx44f K2K_ , cl_mem depthmap_ ){																		// NB must be used _before_ initializing the new cost_volume, because it uses keyframe_imgmem.
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap(..)_chk0 ."<<flush;}
	cl_event writeEvt;
	cl_int status;
	float K2K_arry[16]; for (int i=0; i<16;i++){ K2K_arry[i] = K2K_.operator()(i/4,i%4); }
	
	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * sizeof(float), K2K_arry, 		0, NULL, &writeEvt);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::transform_depthmap(..)_chk1\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	
	cl_int 				res;
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                        	//__private	    uint	    layer,		                    //0
	res = clSetKernelArg(transform_depthmap_kernel,  1, sizeof(cl_mem), &mipmap_buf);			if(res!=CL_SUCCESS){cout<<"\nmipmap_buf = "			<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	                //1
	res = clSetKernelArg(transform_depthmap_kernel,  2, sizeof(cl_mem), &uint_param_buf);		if(res!=CL_SUCCESS){cout<<"\nuint_param_buf = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(transform_depthmap_kernel,  3, sizeof(cl_mem), &k2kbuf);				if(res!=CL_SUCCESS){cout<<"\nk2kbuf = "				<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//3
	res = clSetKernelArg(transform_depthmap_kernel,  3, sizeof(cl_mem), &keyframe_imgmem);		if(res!=CL_SUCCESS){cout<<"\nkeyframe_basemem = "	<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		keyframe_imgmem,				//4
	res = clSetKernelArg(transform_depthmap_kernel,  5, sizeof(cl_mem), &keyframe_depth_mem);	if(res!=CL_SUCCESS){cout<<"\nkeyframe_depth_mem = "	<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		keyframe_depth_mem,				//5
	res = clSetKernelArg(transform_depthmap_kernel,  6, sizeof(cl_mem), &depth_mem);			if(res!=CL_SUCCESS){cout<<"\ndepth_mem = "			<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		depth_mem,						//6
	
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk1 ."<<flush;}
	mipmap_call_kernel( transform_depthmap_kernel, m_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk3 ."<<flush;}
	clFlush(m_queue); status = clFinish(m_queue);																							if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::transform_depthmap(..)_transform_depthmap_kernel\n" << flush;exit_(status);}
	
	status = clEnqueueCopyBuffer( m_queue,  depth_mem,	 keyframe_depth_mem, 		0, 0, mm_size_bytes_C1, 0, NULL, &writeEvt);			if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::transform_depthmap(..)_clEnqueueCopyBuffer\n" << flush;exit_(status);}
	
	clFlush(m_queue); status = clFinish(m_queue);																							if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::transform_depthmap(..)_clfinish_clEnqueueCopyBuffer\n" << flush;exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap(..)_finished ."<<flush;}
}

void RunCL::initializeDepthCostVol( cl_mem key_frame_depth_map_src){			 															// Uses the current frame as the keyframe for a new depth cost volume.
																																			// Dynamic_slam::initialize_from_GT(), Dynamic_slam::initialize_new_keyframe();
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk0 ."<<flush;}
	costvol_frame_num = 0;
	cl_event writeEvt, ev;																													// Load keyframe
	cl_int res, status;
	status = clEnqueueCopyBuffer( m_queue, imgmem, 					keyframe_imgmem, 			0, 0, mm_size_bytes_C4, 0, NULL, &writeEvt);	if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_keyframe_imgmem\n" 	<< flush;exit_(status);}
	clFlush(m_queue); status = clFinish(m_queue);																							if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_clFinish(m_queue)_keyframe_imgmem\n" 	<< flush;exit_(status);}
	
	//key_frame_depth_map_src
	//keyframe_depth_mem
	stringstream ss;
	ss << "__buildDepthCostVol";
	ss << (keyFrameCount*1000 + costVolCount);	
	
	DownloadAndSave(		 	key_frame_depth_map_src,   	ss.str(),   paths.at("key_frame_depth_map_src"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);	cout << "\nDownloadAndSave (.. key_frame_depth_map_src ..) finished\n"<<flush;

	status = clEnqueueCopyBuffer( m_queue, key_frame_depth_map_src, keyframe_depth_mem,			0, 0, mm_size_bytes_C1, 0, NULL, &writeEvt);	if(status!= CL_SUCCESS){cout << "\n status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_key_frame_depth_map_src\n" 				<< flush;exit_(status);}	
	status = clEnqueueCopyBuffer( m_queue, g1mem, 					keyframe_g1mem, 			0, 0, mm_size_bytes_C1, 0, NULL, &writeEvt);	if(status!= CL_SUCCESS){cout << "\n status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_keyframe_g1mem\n" 						<< flush;exit_(status);}
	status = clEnqueueCopyBuffer( m_queue, SE3_grad_map_mem, 		keyframe_SE3_grad_map_mem, 	0, 0, mm_size_bytes_C1, 0, NULL, &writeEvt);	if(status!= CL_SUCCESS){cout << "\n status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_keyframe_SE3_grad_map_mem\n" 			<< flush;exit_(status);}
	clFlush(m_queue); status = clFinish(m_queue);																												if(status!= CL_SUCCESS){cout << "\n status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_clFinish(m_queue)\n" 	<< flush;exit_(status);}
	
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1 ."<<flush;}
	float depth = 1/( obj["max_depth"].asFloat() - obj["min_depth"].asFloat() );															// Zero the new cost vol. NB 'depth' _might_ be a useful start value when bootstrapping.
	float zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, cdatabuf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.1\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, hdatabuf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.2\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, img_sum_buf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.3\n"<< endl;exit_(status);}		clFlush(uload_queue); status = clFinish(uload_queue);
	
	//status = clEnqueueFillBuffer(uload_queue, dmem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.4\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	//status = clEnqueueFillBuffer(uload_queue, amem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, qmem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, lomem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.7\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, himem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	
	clFlush(uload_queue); status = clFinish(uload_queue); 																					if (status != CL_SUCCESS)	{ cout << "\nclFinish(uload_queue)=" << status << checkerror(status) <<"\n"  << flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.5 ."<<flush;}
																																			
																																			if(verbosity>local_verbosity_threshold) {
																																				stringstream ss;
																																				ss << "initializeDepthCostVol";
																																				ss << (keyFrameCount*1000 + costVolCount);													// Save buffers to file ###########
																																				//keyframe_imgmem
																																				cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.5.1 ."<<flush;
																																				DownloadAndSave_3Channel(	keyframe_imgmem,  			ss.str(), paths.at("keyframe_imgmem"), 				mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	false );								cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.6 ."<<flush;
																																				DownloadAndSave(		 	keyframe_depth_mem,   		ss.str(), paths.at("keyframe_depth_mem"),   		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.7 ."<<flush;
																																				DownloadAndSave(		 	keyframe_g1mem,   			ss.str(), paths.at("keyframe_g1mem"),   			mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.8 ."<<flush;
																																				DownloadAndSave(		 	keyframe_SE3_grad_map_mem,  ss.str(), paths.at("keyframe_SE3_grad_map_mem"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk2 ."<<flush;}
}

void RunCL::updateDepthCostVol(cv::Matx44f K2K_, int count, uint start, uint stop){ //buildDepthCostVol();
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateDepthCostVol(..)_chk0 ."<<flush;}
	cl_event writeEvt;
	cl_int status;
	float K2K_arry[16]; for (int i=0; i<16;i++){ K2K_arry[i] = K2K_.operator()(i/4,i%4); }
	
	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * sizeof(float), K2K_arry, 		0, NULL, &writeEvt);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::updateDepthCostVol(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateDepthCostVol(..)_chk0.7 "<<flush;}
	cl_int 				res;
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                        //__private	    uint	    layer,		                    //0
	res = clSetKernelArg(depth_cost_vol_kernel,  1, sizeof(cl_mem), &mipmap_buf);			if(res!=CL_SUCCESS){cout<<"\nmipmap_buf = "			<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	                //1
	res = clSetKernelArg(depth_cost_vol_kernel,  2, sizeof(cl_mem), &uint_param_buf);		if(res!=CL_SUCCESS){cout<<"\nuint_param_buf = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(depth_cost_vol_kernel,  3, sizeof(cl_mem), &fp32_param_buf);		if(res!=CL_SUCCESS){cout<<"\nfp32_param_buf = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//3
	res = clSetKernelArg(depth_cost_vol_kernel,  4, sizeof(cl_mem), &k2kbuf);				if(res!=CL_SUCCESS){cout<<"\nk2kbuf = "				<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//4
	res = clSetKernelArg(depth_cost_vol_kernel,  5, sizeof(cl_mem), &keyframe_imgmem);		if(res!=CL_SUCCESS){cout<<"\nkeyframe_basemem = "	<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//5		// equivalent to basemem
	
	res = clSetKernelArg(depth_cost_vol_kernel,  6, sizeof(cl_mem), &imgmem);				if(res!=CL_SUCCESS){cout<<"\nimgmem = "	<<checkerror(res)<<"\n"<<flush;exit_(res);}					//__global 		float4*		imgmem,							//6		// equivalent to imgmem
	res = clSetKernelArg(depth_cost_vol_kernel,  7, sizeof(cl_mem), &cdatabuf);				if(res!=CL_SUCCESS){cout<<"\ncdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);}		// cdata
	res = clSetKernelArg(depth_cost_vol_kernel,  8, sizeof(cl_mem), &hdatabuf);				if(res!=CL_SUCCESS){cout<<"\nhdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);}		// hdata
	res = clSetKernelArg(depth_cost_vol_kernel,  9, sizeof(cl_mem), &lomem);				if(res!=CL_SUCCESS){cout<<"\nlomem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);}		// lo
	res = clSetKernelArg(depth_cost_vol_kernel, 10, sizeof(cl_mem), &himem);				if(res!=CL_SUCCESS){cout<<"\nhimem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);}		// hi
	res = clSetKernelArg(depth_cost_vol_kernel, 11, sizeof(cl_mem), &amem);					if(res!=CL_SUCCESS){cout<<"\namem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);}		// a	//__global 		float* apt,	// amem, auxilliary A
	res = clSetKernelArg(depth_cost_vol_kernel, 12, sizeof(cl_mem), &dmem);					if(res!=CL_SUCCESS){cout<<"\ndmem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);}		// d	//__global 		float* dpt,	// dmem, depth D
	res = clSetKernelArg(depth_cost_vol_kernel, 13, sizeof(cl_mem), &img_sum_buf);			if(res!=CL_SUCCESS){cout<<"\nimg_sum_buf res = " 	<<checkerror(res)<<"\n"<<flush;exit_(res);}		// cdata
	
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
	
	// ? use epth_mem[!frame_bool_idx]/[frame_bool_idx]  ?  Also for other params ?
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
	mipmap_call_kernel( depth_cost_vol_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateDepthCostVol(..)_chk3 ."<<flush;}
																																			if(verbosity>local_verbosity_threshold) {
																																				stringstream ss;
																																				ss << "buildDepthCostVol";
																																				ss << (keyFrameCount*1000 + costVolCount);													// Save buffers to file ###########
																																				DownloadAndSave_3Channel(	imgmem,  			ss.str(), paths.at("imgmem"), 		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	false );
																																				DownloadAndSave(			lomem,  			ss.str(), paths.at("lomem"),  		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , 8); // a little more than the num images in costvol.
																																				DownloadAndSave(		 	himem,  			ss.str(), paths.at("himem"),  		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , 8); //params[LAYERS]
																																				DownloadAndSave(		 	amem,   			ss.str(), paths.at("amem"),   		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																																				DownloadAndSave(		 	dmem,   			ss.str(), paths.at("dmem"),   		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																																				DownloadAndSaveVolume(		cdatabuf, 			ss.str(), paths.at("cdatabuf"), 	mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  false , 1);
																																				if(verbosity>1) cout << "\ncostVolCount="<<costVolCount;
																																				cout << "\nRunCL::updateDepthCostVol(..)_finished\n" << flush;
																																			}
}

void RunCL::updateQD(float epsilon, float theta, float sigma_q, float sigma_d, int count, uint start, uint stop){
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateQD(..)_chk0 ."<<flush;}
	key_frame_QD_num++;
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

	stringstream ss;
	ss << "updateQD";
	
	// __private	uint	layer	, set in mipmap_call_kernel(..) below																									//__private	uint	layer,				//0
	res = clSetKernelArg(updateQD_kernel, 1, sizeof(cl_mem), &mipmap_buf);		if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__constant 	uint*	mipmap_params,	//1
	res = clSetKernelArg(updateQD_kernel, 2, sizeof(cl_mem), &uint_param_buf);	if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__constant 	uint*	uint_params,	//2
	res = clSetKernelArg(updateQD_kernel, 3, sizeof(cl_mem), &fp32_param_buf);	if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global 	float*  fp32_params,		//3
	res = clSetKernelArg(updateQD_kernel, 4, sizeof(cl_mem), &keyframe_g1mem);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res) <<"\n"<<flush;exit_(res);}			//__global 	float* 	g1pt,				//4
	res = clSetKernelArg(updateQD_kernel, 5, sizeof(cl_mem), &qmem);	 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res) <<"\n"<<flush;exit_(res);}			//__global 	float* 	qpt,				//5
	res = clSetKernelArg(updateQD_kernel, 6, sizeof(cl_mem), &amem);	 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res) <<"\n"<<flush;exit_(res);}			//__global 	float*  apt,				//6		// amem,     auxilliary A
	res = clSetKernelArg(updateQD_kernel, 7, sizeof(cl_mem), &dmem);	 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res) <<"\n"<<flush;exit_(res);}			//__global 	float*  dpt					//7		// dmem,     depth D
	
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateQD(..)_chk1 ."<<flush;}
	mipmap_call_kernel( updateQD_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateQD(..)_chk3 ."<<flush;}
																																			if(verbosity>local_verbosity_threshold){
																																				//size_t st  = width * height * sizeof(float);
																																				QDcount++;
																																				int this_count = count + QDcount;
																																				ss << this_count;
																																				cv::Size q_size( mm_Image_size.width, 2* mm_Image_size.height ); // 2x sized for qx and qy.
																																				//cout<<"\n\nRunCL::updateQD(..)_chk3.1 ."<<flush;
																																				DownloadAndSave(qmem,   ss.str(), paths.at("qmem"),  2*mm_size_bytes_C1 , q_size        , CV_32FC1, false , -1*fp32_params[MAX_INV_DEPTH]  );
																																				//cout<<"\n\nRunCL::updateQD(..)_chk3.2 ."<<flush;
																																				DownloadAndSave(dmem,   ss.str(), paths.at("dmem"),    mm_size_bytes_C1 , mm_Image_size , CV_32FC1, false ,    fp32_params[MAX_INV_DEPTH]  );
																																				
																																				cout<<"\nRunCL::updateQD_chk3_finished\n"<<flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateQD(..)_finished ."<<flush;}
}

void RunCL::updateG(int count, uint start, uint stop){
	int local_verbosity_threshold = 0;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateG(..)_chk0"<<flush;}
	key_frame_cacheG_num++;
	cl_int res;
	size_t num_threads = ceil( (float)(mm_layerstep)/(float)local_work_size ) * local_work_size ; 
																																			if(verbosity>local_verbosity_threshold) { cout<<"\n\nRunCL::updateG(..)_chk1"<<flush;
																																				cout << ",   num_threads = " << num_threads << ",   mm_layerstep = " << mm_layerstep << ",  local_work_size = " << local_work_size  <<endl << flush;}
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                            //__private	 uint	    layer,			//0
    res = clSetKernelArg(updateG_kernel, 1, sizeof(cl_mem), &mipmap_buf);						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*		mipmap_params,	//1
	res = clSetKernelArg(updateG_kernel, 2, sizeof(cl_mem), &uint_param_buf);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*		uint_params		//2
	res = clSetKernelArg(updateG_kernel, 3, sizeof(cl_mem), &fp32_param_buf);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant float*		fp32_params		//3
	res = clSetKernelArg(updateG_kernel, 4, sizeof(cl_mem), &keyframe_imgmem);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global   float4*	img,			//4
	res = clSetKernelArg(updateG_kernel, 5, sizeof(cl_mem), &keyframe_g1mem);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	g1p				//5
	
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateG(..)_chk2"<<flush;}
	mipmap_call_kernel( updateG_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateG(..)_chk3 Saving keyframe_g1mem."<<flush;
																																				stringstream ss;	ss << frame_num << "_updateG";
																																				stringstream ss_path;	
																																				ss_path << "keyframe_g1mem"; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_3Channel(	keyframe_g1mem, ss.str(), paths.at(ss_path.str()),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateG(..)_chk4 Finished."<<flush;}
}

void RunCL::updateA(float lambda, float theta, int count, uint start, uint stop){
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateA(..)_chk0 ."<<flush;}
	fp32_params[THETA]			=  theta;
	fp32_params[LAMBDA]			=  lambda;
	cl_int status;
	cl_event writeEvt;
	status = clEnqueueWriteBuffer(uload_queue,  fp32_param_buf, CL_FALSE, 0, 16 * sizeof(float), fp32_params, 0, NULL, &writeEvt);										// WriteBuffer param_buf ##########
												if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: \nRunCL::updateA_chk0\n" << endl; exit_(status);}
																																			else if(verbosity>0) {cout << "\nRunCL::updateA_chk0.5\t\tlayers="<< fp32_params[LAYERS] <<" \n" << flush;}
																																			
	status = clFlush(uload_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush status = " << status << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = waitForEventAndRelease(&writeEvt); if (status != CL_SUCCESS)	{ cout << "\nwaitForEventAndRelease status = "<<status<<checkerror(status)<<"\n"<<flush; exit_(status);}

	stringstream ss;
	ss << "updateA";
	cl_int res;
	cl_event ev;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateA(..)_chk1 ."<<flush;}
	
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                            //__private	 uint	    layer,			//0
	res = clSetKernelArg(updateA_kernel, 1, sizeof(cl_mem), &mipmap_buf); 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant 	uint*	mipmap_params,	//1
	res = clSetKernelArg(updateA_kernel, 2, sizeof(cl_mem), &uint_param_buf); 	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant 	uint*	uint_params,	//2
	res = clSetKernelArg(updateA_kernel, 3, sizeof(cl_mem), &fp32_param_buf); 	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  fp32_params,		//3
	res = clSetKernelArg(updateA_kernel, 4, sizeof(cl_mem), &cdatabuf); 		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  cdata,				//4		//           cost volume
	res = clSetKernelArg(updateA_kernel, 7, sizeof(cl_mem), &lomem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  lo,					//5
	res = clSetKernelArg(updateA_kernel, 8, sizeof(cl_mem), &himem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  hi,					//6
	res = clSetKernelArg(updateA_kernel, 5, sizeof(cl_mem), &amem);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  apt,				//7		// amem,     auxilliary A
	res = clSetKernelArg(updateA_kernel, 6, sizeof(cl_mem), &dmem);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float*  dpt					//8		// dmem,     depth D

	status = clFlush(m_queue); 					if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateA(..)_chk2 ."<<flush;}
	mipmap_call_kernel( updateA_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::updateA(..)_chk3 ."<<flush;}
	count = keyFrameCount*1000000 + A_count*1000 + 999;
	A_count++;
											if(A_count%1==0 && verbosity>local_verbosity_threshold){
												ss << count << "_theta"<<theta<<"_";
												cv::Size q_size( mm_Image_size.width, 2* mm_Image_size.height );
												DownloadAndSave(amem,   ss.str(), paths.at("amem"),    mm_size_bytes_C1,   mm_Image_size, CV_32FC1,  false , fp32_params[MAX_INV_DEPTH]);
												DownloadAndSave(dmem,   ss.str(), paths.at("dmem"),    mm_size_bytes_C1,   mm_Image_size, CV_32FC1,  false , fp32_params[MAX_INV_DEPTH]);
												DownloadAndSave(qmem,   ss.str(), paths.at("qmem"),  2*mm_size_bytes_C1,   q_size       , CV_32FC1,  false , 0.1 );
											}
																																			if(verbosity>0) cout<<"\nRunCL::updateA_chk2_finished,   fp32_params[MAX_INV_DEPTH]="<<fp32_params[MAX_INV_DEPTH]<<flush;
}



























/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void RunCL::SpatialCostFns(){ //SpatialCostFns();


}

void RunCL::ParsimonyCostFns(){ //ParsimonyCostFns();


}


void RunCL::ExhaustiveSearch(){ //ExhaustiveSearch();

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
    
    
