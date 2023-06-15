#include "RunCL.h"

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

using namespace std;

void RunCL::predictFrame(){ //predictFrame();


}

void RunCL::loadFrame(cv::Mat image){ //getFrame();
	int local_verbosity_threshold = 1;
                                                                                                                                            if(verbosity>0) {cout << "\n RunCL::loadFrame_chk 0\n" << flush;}
	cl_int status;
	cl_event writeEvt;																										               // WriteBuffer basemem #########
	status = clEnqueueWriteBuffer(uload_queue, basemem, CL_FALSE, 0, image_size_bytes, image.data, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
                                                                                                                                            if (verbosity>local_verbosity_threshold){
                                                                                                                                                stringstream ss;	ss << frame_num;
                                                                                                                                                DownloadAndSave_3Channel(basemem, ss.str(), paths.at("basemem"), image_size_bytes, baseImage_size,  baseImage_type, 	false );
                                                                                                                                            }
}

void RunCL::cvt_color_space(){ //getFrame(); basemem(CV_8UC3, RGB)->imgmem(CV16FC3, HSV), NB we will use basemem for image upload, and imgmem for the MipMap. RGB is default for .png standard.
	int local_verbosity_threshold = 1;
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
	res = clSetKernelArg(cvt_color_space_linear_kernel, 1, sizeof(cl_mem), &imgmem[frame_bool_idx]);	   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global float4*		img,			//1	
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
                                                                                                                                                stringstream ss_path;	ss_path << "imgmem[" << frame_bool_idx << "]";
                                                                                                                                                
                                                                                                                                                cv::Size new_Image_size = cv::Size(mm_width, mm_height);
                                                                                                                                                size_t   new_size_bytes = mm_width * mm_height * 4* 4;
                                                                                                                                                
                                                                                                                                                cout << "imgmem[frame_bool_idx]="<< imgmem[frame_bool_idx] << endl << flush;
                                                                                                                                                cout <<", ss.str()="<< ss.str() << endl << flush;
                                                                                                                                                cout <<", paths.at(\"imgmem[0]\")="<< paths.at("imgmem[0]") << endl << flush;
                                                                                                                                                cout <<", paths.at(\"imgmem[1]\")="<< paths.at("imgmem[1]") << endl << flush;
                                                                                                                                                //cout <<", paths.at(\"imgmem[frame_bool_idx]\")="<< paths.at("imgmem[frame_bool_idx]") << endl << flush;
                                                                                                                                                
                                                                                                                                                cout <<", paths.at(" << ss_path.str() <<")="<< paths.at(ss_path.str()) << endl << flush;
                                                                                                                                                cout <<", new_size_bytes="<< new_size_bytes << endl << flush;
                                                                                                                                                cout <<", new_Image_size="<< new_Image_size <<"" << endl << flush;
                                                                                                                                                
                                                                                                                                                DownloadAndSave_3Channel(	imgmem[frame_bool_idx], ss.str(), paths.at( ss_path.str() ), new_size_bytes/*mm_size_bytes_C4*/, new_Image_size/*mm_Image_size*/,  CV_32FC4 /*mm_Image_type*/, 	false );
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
																																			if(verbosity>local_verbosity_threshold){
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
	res = clSetKernelArg(img_variance_kernel, 1, sizeof(cl_mem), &imgmem[frame_bool_idx]);  			   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global float4*		img,			//1	
	res = clSetKernelArg(img_variance_kernel, 2, sizeof(cl_mem), &uint_param_buf);						   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uint*		uint_params		//2
	res = clSetKernelArg(img_variance_kernel, 3, sizeof(cl_mem), &mipmap_buf);							   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__constant uint*		mipmap_params,	//3 // NB layer = 0.
	res = clSetKernelArg(img_variance_kernel, 4, local_work_size*4*sizeof(float), 	NULL);				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		local_sum_pix	//4
	res = clSetKernelArg(img_variance_kernel, 5, sizeof(cl_mem), &var_sum_mem);							   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		global_sum_pix	//5
	
	status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk1,  global_work_size="<< global_work_size <<flush;
	res = clEnqueueNDRangeKernel(m_queue, img_variance_kernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, &ev); 			// run img_variance _kernel  aka img_variance(..) ##### TODO which CommandQueue to use ? What events to check ?
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

/*
#define MiM_PIXELS			0	// for mipmap_buf, 				when launching one kernel per layer. 	Updated for each layer.
#define MiM_READ_OFFSET		1	// for ths layer, 				start of image data
#define MiM_WRITE_OFFSET	2
#define MiM_READ_COLS		3	// cols without margins
#define MiM_WRITE_COLS		4
#define MiM_GAUSSIAN_SIZE	5	// filter box size
#define MiM_READ_ROWS		6	// rows without margins
#define MiM_WRITE_ROWS		7
*/

void RunCL::mipmap_linear(){
	int local_verbosity_threshold = 1;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap(..)_chk0"<<flush;}
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
    res = clSetKernelArg(mipmap_linear_kernel, 1, sizeof(cl_mem), 					 	&mipmap_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		mipmap_params,	//1
	res = clSetKernelArg(mipmap_linear_kernel, 2, sizeof(cl_mem), 					 	&gaussian_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant float*		gaussian,		//2
	res = clSetKernelArg(mipmap_linear_kernel, 3, sizeof(cl_mem), 					 	&uint_param_buf);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		uint_params,	//3
	res = clSetKernelArg(mipmap_linear_kernel, 4, sizeof(cl_mem), 						&imgmem[frame_bool_idx]);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global   float4*	img,			//4	
	res = clSetKernelArg(mipmap_linear_kernel, 5, (local_size+4) *5*4* sizeof(float), 	NULL);						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__local    float4*	local_img_patch //5
	
	mipmap_call_kernel( mipmap_linear_kernel, m_queue, 0, 5 );// TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.
	
	/*
	uint read_rows				= baseImage_height;
	uint write_rows 			= read_rows/2;
	uint margin					= mm_margin;
	uint read_cols_with_margin 	= mm_width ;
	uint read_rows_with_margin	= read_rows + margin;
	mipmap[MiM_READ_OFFSET] 	= margin*mm_width + margin;
	mipmap[MiM_WRITE_OFFSET] 	= read_cols_with_margin * read_rows_with_margin + mipmap[MiM_READ_OFFSET];
	mipmap[MiM_READ_COLS] 		= baseImage_width;
	mipmap[MiM_WRITE_COLS] 		= mipmap[MiM_READ_COLS]/2;
	mipmap[MiM_GAUSSIAN_SIZE] 	= mm_gaussian_size;
																																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap(..)_chk1"<<flush;}
	for(int reduction = 0; reduction < mm_num_reductions; reduction++) {
																																															if(verbosity>local_verbosity_threshold) {
																																																cout<<"\n\nRunCL::mipmap(..)_chk2"<<flush;
																																																cout << "\nreduction="<< reduction << " , read_rows=" << read_rows  << " ,  write_rows=" <<  write_rows  << " ,  read_cols_with_margin=" << 	read_cols_with_margin  << " ,  read_rows_with_margin=" <<  read_rows_with_margin  << " ,  margin=" << 	margin  << " ,   mipmap[MiM_READ_OFFSET]=" <<  mipmap[MiM_READ_OFFSET]  << " ,  mipmap[MiM_WRITE_OFFSET]=" <<  mipmap[MiM_WRITE_OFFSET]	  << " ,  mipmap[MiM_READ_COLS]=" <<   mipmap[MiM_READ_COLS]  << " ,   mipmap[MiM_WRITE_COLS]=" <<    mipmap[MiM_WRITE_COLS]  << " ,   mipmap[MiM_GAUSSIAN_SIZE]=" <<    mipmap[MiM_GAUSSIAN_SIZE] << endl << flush; 
																																															}
		mipmap[MiM_PIXELS]		= write_rows*mipmap[MiM_WRITE_COLS];																														// compute num threads to launch & num_pixels in reduction
		size_t num_threads		= ceil( (float)(mipmap[MiM_PIXELS])/(float)local_work_size ) * local_work_size ;																			// global_work_size formula  
																																															// write mipmap_buf
		status = clEnqueueWriteBuffer(uload_queue, mipmap_buf, 	CL_FALSE, 0, 8 * sizeof(uint), 	mipmap, 0, NULL, &writeEvt);	
		if (status != CL_SUCCESS){cout<<"\nstatus = "<<checkerror(status)<<"\n"<<flush; cout << "Error: RunCL::mipmap, clEnqueueWriteBuffer, mipmap_buf \n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
		res = clSetKernelArg(mipmap_linear_kernel, 3, sizeof(cl_mem), &mipmap_buf);	
											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;																//__global uint*	mipmap_params	//3
		status = clFlush(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}						// clEnqueueNDRangeKernel
		status = clFinish(m_queue); 		if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
		
		res = clEnqueueNDRangeKernel(m_queue, mipmap_linear_kernel, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev); 																	// run mipmap_linear_kernel, NB wait for own previous iteration.
		if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		status = clFlush(m_queue);			if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clWaitForEvents (1, &ev);	if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}		// update read&write rows&cols
		
		mipmap[MiM_READ_OFFSET] 	= mipmap[MiM_WRITE_OFFSET];
		mipmap[MiM_WRITE_OFFSET] 	= mipmap[MiM_WRITE_OFFSET] + read_cols_with_margin * (margin + write_rows);
		read_rows					= margin + write_rows;
		write_rows					= write_rows/2;
		mipmap[MiM_READ_COLS] 		= mipmap[MiM_WRITE_COLS];
		mipmap[MiM_WRITE_COLS] 		= mipmap[MiM_WRITE_COLS]/2;
		mipmap[MiM_PIXELS]			= mipmap[MiM_WRITE_COLS] * write_rows;
																																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap(..)_chk2.6 Finished one loop"<<flush;}
	}
	*/
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << frame_num << "_mipmap";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				stringstream ss_path;	ss_path << "imgmem[" << frame_bool_idx << "]";
																																				DownloadAndSave_3Channel( imgmem[frame_bool_idx], ss.str(), paths.at(ss_path.str()), new_size_bytes, new_Image_size, CV_32FC4, false );
																																				cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap(..)_chk4 Finished"<<flush;}
}


void RunCL::img_gradients(){ //getFrame();
	int local_verbosity_threshold = 1;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk0"<<flush;}
	cl_int res;
	size_t num_threads = ceil( (float)(mm_layerstep)/(float)local_work_size ) * local_work_size ; 
																																			if(verbosity>local_verbosity_threshold) {cout << "\n num_threads = " << num_threads << ",   mm_layerstep = " << mm_layerstep << ",  local_work_size = " << local_work_size  <<endl << flush;}
	//res = clSetKernelArg(img_grad_kernel, 3, sizeof(cl_mem), &mipmap_buf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global uint*	mipmap_params,	//3
	
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	 uint	    layer,		//0
    res = clSetKernelArg(img_grad_kernel, 1, sizeof(cl_mem), &mipmap_buf);				                      if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*	mipmap_params,	//1
	res = clSetKernelArg(img_grad_kernel, 2, sizeof(cl_mem), &uint_param_buf);						          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*	uint_params		//2
	res = clSetKernelArg(img_grad_kernel, 3, sizeof(cl_mem), &fp32_param_buf);						          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant float*	fp32_params		//3
	res = clSetKernelArg(img_grad_kernel, 4, sizeof(cl_mem), &imgmem[frame_bool_idx]);				          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global   float4*	img,		//4
	res = clSetKernelArg(img_grad_kernel, 5, sizeof(cl_mem), &gxmem[frame_bool_idx]);				          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	gxp,		//5
	res = clSetKernelArg(img_grad_kernel, 6, sizeof(cl_mem), &gymem[frame_bool_idx]);				          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	gyp,		//6
	res = clSetKernelArg(img_grad_kernel, 7, sizeof(cl_mem), &g1mem[frame_bool_idx]);				          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	g1p			//7
	res = clSetKernelArg(img_grad_kernel, 8, sizeof(cl_mem), &SE3_map_mem);							          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant float2*	SE3_map,	//8
	res = clSetKernelArg(img_grad_kernel, 9, sizeof(cl_mem), &SE3_grad_map_mem[frame_bool_idx]);	          if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	SE3_grad_map//9
	
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk2"<<flush;}
	mipmap_call_kernel( img_grad_kernel, m_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk3 Finished all loops. Saving gxmem, gymem, g1mem."<<flush;
																																				stringstream ss;	ss << frame_num << "_img_grad";
																																				stringstream ss_path;	
																																				ss_path << "gxmem[" << frame_bool_idx << "]"; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_3Channel(	gxmem[frame_bool_idx], ss.str(), paths.at(ss_path.str()),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "gymem[" << frame_bool_idx << "]"; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_3Channel(	gymem[frame_bool_idx], ss.str(), paths.at(ss_path.str()),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "g1mem[" << frame_bool_idx << "]"; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_3Channel(	g1mem[frame_bool_idx], ss.str(), paths.at(ss_path.str()),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				///
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "SE3_grad_map_mem[" << frame_bool_idx << "]"<<flush; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_6Channel_volume(  SE3_grad_map_mem[frame_bool_idx], ss.str(), paths.at(ss_path.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																				///
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "SE3_grad_map_mem[!" << frame_bool_idx << "]"<<flush; 
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_6Channel_volume(  SE3_grad_map_mem[!frame_bool_idx], ss.str(), paths.at(ss_path.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																				//
																																				cout << "\n\n SE3_grad_map_mem[frame_bool_idx] = SE3_grad_map_mem["<<frame_bool_idx<<"] = "<<SE3_grad_map_mem[frame_bool_idx];
																																				
                                                                                                                                                cout << "\n SE3_grad_map_mem[!frame_bool_idx] = SE3_grad_map_mem["<<!frame_bool_idx<<"] = "<<SE3_grad_map_mem[!frame_bool_idx]<<endl<<flush;
																																				
																																				//DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers )
																																				//SE3_grad_map_mem[frame_bool_idx]  // SE3_grad_map[read_index + i* mm_pixels]
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk4 Finished."<<flush;}
}

void RunCL::loadFrameData(cv::Mat GT_depth, cv::Matx44f GT_K2K,   cv::Matx44f GT_pose2pose){ //getFrameData();
    int local_verbosity_threshold = 1;
	
    for (int i=0; i<16; i++){ fp32_k2k[i] = GT_K2K.operator()(i/4, i%4);   																if(verbosity>local_verbosity_threshold+2) cout << "\nK2K ("<<i%4 <<","<< i/4<<") = "<< fp32_k2k[i]; }
    
    cl_event 			writeEvt;
	cl_int 				status;
    status = clEnqueueWriteBuffer(uload_queue, depth_mem, 		CL_FALSE, 0, mm_size_bytes_C1,	 GT_depth.data, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
    
}

void RunCL::precom_param_maps(float SE3_k2k[6*16]){ //  Compute maps of pixel motion for each SE3 DoF, and camera params // Derived from RunCL::mipmap
	int local_verbosity_threshold = 1;
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

void RunCL::estimateSO3(uint start, uint stop){ //estimateSO3();	(uint start=0, uint stop=8)


}

void RunCL::estimateSE3(uint start, uint stop){ //estimateSE3(); 	(uint start=0, uint stop=8)			// TODO replace arbitrary fixed constant with a const uint variable in the header...
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk0 ."<<flush;}
    cl_event writeEvt;
    cl_int status;
	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * sizeof(float), fp32_k2k, 		0, NULL, &writeEvt);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
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
	res = clSetKernelArg(se3_grad_kernel,10, local_work_size*6*4*sizeof(float), NULL);					        if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_grads					//10	6 DoF, float4 channels
	res = clSetKernelArg(se3_grad_kernel,11, sizeof(cl_mem), &se3_sum_mem);		 					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		g1p,							//11
	res = clSetKernelArg(se3_grad_kernel,12, sizeof(cl_mem), &SE3_incr_map_mem);					            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_incr_map_					//12
	res = clSetKernelArg(se3_grad_kernel,13, sizeof(cl_mem), &SE3_rho_map_mem);					                if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     rho_					        //13
	
																																			if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::estimateSE3(..)_chk1 ."<<flush;}
	mipmap_call_kernel( se3_grad_kernel, m_queue, start, stop );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << frame_num << "_img_grad";
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
                                                                                                                                                DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at(ss_path_rho.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 1 );
                                                                                                                                                
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
    cv::Mat se3_sum_mat = cv::Mat::zeros (se3_sum_size, 6*4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)					// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
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
    float SE3_reults[8][6][4] = {{{0}}}; 																									// max 8 layers, 6 DoF, 4 channels
																																			if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::estimateSE3(..)_chk6 ."<<flush;
																																				cout << "\n\nse3_sum_mat.at<float> (i*6 + j,  k) ";
																																				for (int i=0; i< se3_sum_size ; i++){
																																					cout << "\ni="<<i<<":   ";
																																					for (int j=0; j<6; j++){
																																						cout << ",     (";
																																						for (int k=0; k<4; k++){
																																							cout << "," << se3_sum_mat.at<float> (i , j*4 + k)  ;
																																						}cout << ")";
																																					}cout << flush;
																																				}
																																				cout << endl << endl;
																																			}
	
	for (int i=0; i<=mm_num_reductions+1; i++){ 
        uint read_offset_ 	= MipMap[i*8 + MiM_READ_OFFSET];                                                                                // mipmap_params_[MiM_READ_OFFSET];
        uint global_sum_offset = read_offset_ / local_work_size ;
        
        uint groups_to_sum = se3_sum_mat.at<float>(global_sum_offset, 0);
        uint start_group = global_sum_offset + 1;
        uint stop_group = start_group + groups_to_sum ;   // -1                                                                                // skip the last group due to odd 7th value.
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\ni="<<i<<", read_offset_="<<read_offset_<<",  global_sum_offset="<<global_sum_offset<<",  groups_to_sum="<<groups_to_sum<< ",  start_group="<<start_group<<",  stop_group="<<stop_group;
																																			}
        for (int j=start_group; j< stop_group  ; j++){
            for (int k=0; k<6; k++){
				for (int l=0; l<4; l++){
					SE3_reults[i][k][l] += se3_sum_mat.at<float>(j, k*4 + l); // se3_sum_mat.at<float>(j, k);                         		// sum j groups for this layer of the MipMap.
				}
            }
        }
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\nLayer "<<i<<" SE3_results = (";																// raw results
																																				for (int k=0; k<6; k++){
																																					cout << "(";
																																					for (int l=0; l<4; l++){
																																						cout << ", " << SE3_reults[i][k][l] ;
																																					}cout << ")";
																																				}cout << ")";
																																			}
    }
																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << endl;
																																				for (int i=0; i<=mm_num_reductions+1; i++){ 															// results / (num_valid_px * img_variance) 
																																					cout << "\nLayer "<<i<<" SE3_results/num_groups = (";
																																					for (int k=0; k<6; k++){
																																						cout << "(";
																																						for (int l=0; l<3; l++){
																																							cout << ", " << SE3_reults[i][k][l] / ( SE3_reults[i][k][3]  *  img_stats[IMG_VAR+l]  );	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}
																																						cout << ", " << SE3_reults[i][k][3] << ")";
																																					}cout << ")";
																																				}
																																			}
    
    
}

/*
{
	//mipmap_call_kernel( reduce_kernel, m_queue, start, stop ); // TODO Might be the wong wrapper function... #######################################  Does it need to be called repeatedly eg for images larger than work_group_size**2 , esp if  work_group_size is small. !!!!!!!!!!
	//###################################################################################################
	
	uint se3_sum_red_params[8][8] = {{0}};
	uint *layer_params;
	
	uint pixels = baseImage_width * baseImage_height;
	se3_sum_size 		= 0;
	for (int i=0; i<mm_num_reductions; i++){
		layer_params = se3_sum_red_params[i];
		se3_sum_size 		+= 1+ ceil( (float)(pixels) / (float)local_work_size ) ;											            // Space for each group of eachlayer, plus layer information. See use between se3_grad(..) and reduction(..) kernels.
		layer_params[0]	= se3_sum_size;
		pixels /= 4;
	}
	se3_sum_size_bytes	= se3_sum_size * sizeof(float) * 8;																	                if(verbosity>local_verbosity_threshold) cout <<"\n\n se3_sum_size="<< se3_sum_size<<",    se3_sum_size_bytes="<<se3_sum_size_bytes<<flush;
	se3_sum2_size_bytes = 2 * mm_num_reductions*sizeof(float)*8;
	
	
	
	// #######################
	uint mipmap_layer_params[8][8] = {{0}};
																																			if(verbosity>local_verbosity_threshold) { cout<<"\n\nRunCL::estimateSE3(..)_chk6, preparing to call reduce(..) kernel"<<flush; }
	cl_event						writeEvt, ev;
	cl_int							status;
	uint 							mipmap[8];
	mipmap[MiM_READ_ROWS] 			= baseImage_height;
	uint write_rows 				= mipmap[MiM_READ_ROWS] /2; 
	uint margin						= mm_margin;
	uint read_cols_with_margin 		= mm_width ;
	uint read_rows_with_margin		= mipmap[MiM_READ_ROWS] + margin;  
	mipmap[MiM_READ_OFFSET]			= margin*mm_width + margin;
	
	
	
	mipmap[MiM_WRITE_OFFSET]		= read_cols_with_margin * read_rows_with_margin + mipmap[MiM_READ_OFFSET];
	mipmap[MiM_READ_COLS]			= baseImage_width;
	mipmap[MiM_WRITE_COLS]			= mipmap[MiM_READ_COLS]/2;
	mipmap[MiM_PIXELS]				= mipmap[MiM_READ_COLS] * mipmap[MiM_READ_ROWS];
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk1"<<flush;}
	for(int reduction = 0; reduction <= mm_num_reductions+1; reduction++) {
																																			if(verbosity>local_verbosity_threshold) {
                                                                                                                                                cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2"<<flush;
																																				cout << "\nreduction="<< reduction << " , read_rows=" << mipmap[MiM_READ_ROWS]   << " ,  write_rows=" <<  write_rows  << " ,  read_cols_with_margin=" << 	read_cols_with_margin  << " ,  read_rows_with_margin=" <<  read_rows_with_margin  << " ,  margin=" << 	margin  << " ,   mipmap[MiM_READ_OFFSET]=" <<  mipmap[MiM_READ_OFFSET]  << " ,  mipmap[MiM_WRITE_OFFSET]=" <<  mipmap[MiM_WRITE_OFFSET]	  << " ,  mipmap[MiM_READ_COLS]=" <<   mipmap[MiM_READ_COLS]  << " ,   mipmap[MiM_WRITE_COLS]=" <<    mipmap[MiM_WRITE_COLS]  << " ,   mipmap[MiM_GAUSSIAN_SIZE]=" <<    mipmap[MiM_GAUSSIAN_SIZE] << endl << flush; 
																																			}
		
		
		if (reduction>=start && reduction<stop){																							// compute num threads to launch & num_pixels in reduction
			size_t num_threads		= ceil( (float)(mipmap[MiM_PIXELS])/(float)local_work_size ) * local_work_size ;						// global_work_size formula  
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.1   num_threads="<<num_threads <<",  mipmap[MiM_PIXELS]="<<mipmap[MiM_PIXELS]<<flush;}
																																			// write mipmap_buf
			status 	= clEnqueueWriteBuffer(uload_queue, mipmap_buf, 	CL_FALSE, 0, 8 * sizeof(uint), 	mipmap, 0, NULL, &writeEvt);
                                                                                                                if (status != CL_SUCCESS)	{ cout<<"\nstatus = "<<checkerror(status)<<"\n"<<flush; cout << "Error: RunCL::mipmap_call_kernel, clEnqueueWriteBuffer, mipmap_buf \n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
			res 	= clSetKernelArg(reduce_kernel, 0, sizeof(cl_mem), &mipmap_buf);							if(res!=CL_SUCCESS)			{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global uint*	mipmap_params	//3
			status 	= clFlush(m_queue); 																		if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}	// clEnqueueNDRangeKernel
			status 	= clFinish(m_queue); 																		if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
		// *
		//cl_int clEnqueueNDRangeKernel(
		//								cl_command_queue command_queue,
		//								cl_kernel kernel,
		//								cl_uint work_dim,
		//								const size_t* global_work_offset,
		//								const size_t* global_work_size,
		//								const size_t* local_work_size,
		/								cl_uint num_events_in_wait_list,
		//								const cl_event* event_wait_list,
		//								cl_event* event);
		// * /
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.2 , num_threads="<<num_threads<<", local_work_size="<<local_work_size<<flush;}
			res = clEnqueueNDRangeKernel(m_queue, reduce_kernel, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev); 						// run mipmap_linear_kernel, NB wait for own previous iteration.
                                                                                                                if (res != CL_SUCCESS)		{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		status = clFlush(m_queue);																				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clWaitForEvents (1, &ev);																		if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}		
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.3 "<<flush;}
		}
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.4 "<<flush;}
		mipmap[MiM_READ_OFFSET] 	= mipmap[MiM_WRITE_OFFSET];
		mipmap[MiM_WRITE_OFFSET] 	= mipmap[MiM_WRITE_OFFSET] + read_cols_with_margin * (margin + write_rows);
		mipmap[MiM_READ_ROWS] 		= write_rows;
		write_rows					= write_rows/2;
		mipmap[MiM_READ_COLS] 		= mipmap[MiM_WRITE_COLS];
		mipmap[MiM_WRITE_COLS] 		= mipmap[MiM_WRITE_COLS]/2;
		mipmap[MiM_PIXELS]			= mipmap[MiM_READ_COLS] * mipmap[MiM_READ_ROWS];
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.6 Finished one loop"<<flush;}
	}																																		if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk3 Finished "<<flush;}

	
	
	
	
// #################################################################################################
	// NB :  Num groups = 600, 150, 38, 10, 3, 1. Write to: mm_size_bytes_C1*28, float8 global_sum_grads[group_id] = local_sum_grads[0] / local_sum_grads[0][7]; 
	
	// get result of reduction
	cl_event readEvt;
	//cl_int status;
	float SE3_grad[12]={0};
	status = clEnqueueReadBuffer(	dload_queue,		// command_queue
									se3_sum_mem,		// buffer
									CL_FALSE,			// blocking_read
									0,					// offset
									12*sizeof(float),	// size
									SE3_grad,			// pointer
									0,					// num_events_in_wait_list
									NULL,				// event_waitlist				needs to know about preceeding events:
									&readEvt);			// event
                                                                                                                if (status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer(..) status=" << checkerror(status) <<"\n"<<flush; exit_(status);} 
	status = clFlush(dload_queue);					                                                            if (status != CL_SUCCESS) { cout << "\nclFlush(m_queue) status = " 		<< checkerror(status) <<"\n"<<flush; exit_(status);} 
	status = clWaitForEvents(1, &readEvt); 			                                                            if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status="			<< checkerror(status) <<"\n"<<flush; exit_(status);} 
																																			if (verbosity>local_verbosity_threshold){
																																				cout<<"\n\nRunCL::estimateSE3(..)_chk5 ."<<flush;
																																				cout <<"\n\nSE3_grad[12]=(";
																																				for (int i=0; i<12; i++)   {cout << SE3_grad[i] << ", ";}
																																				cout <<")\n";
																																			}
	
	// set SE3 grad values from result
	
	
}
*/

void RunCL::estimateCalibration(){ //estimateCalibration(); 		// own thread, one iter.


}	

void RunCL::buildDepthCostVol(){ //buildDepthCostVol();


}

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
    
    
