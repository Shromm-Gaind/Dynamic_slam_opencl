#include "RunCL.h"

void RunCL::createFolders(){
																																			if(verbosity>0) cout << "\n createFolders_chk 0\n" << flush;
	std::time_t   result  = std::time(nullptr);
	std::string   out_dir = std::asctime(std::localtime(&result));
	out_dir.pop_back(); 																													// req to remove new_line from end of string.

	boost::filesystem::path 	out_path(boost::filesystem::current_path());
	boost::filesystem::path 	conf_outpath( obj["out_path"].asString() );
																																			cout << "\nconf_outpath = " << conf_outpath ;
	if (conf_outpath.empty()  ) {
		out_path = out_path.parent_path().parent_path();																					// move "out_path" up two levels in the directory tree.
		out_path += conf_outpath;
																																			cout << "  conf_outpath.empty()==true" ;
	}else {out_path = conf_outpath;}
	out_path += "/output/";
																																			cout << "  out_path = " << out_path << endl << flush;

	if(boost::filesystem::create_directory(out_path)) { std::cerr<< "Directory Created: "<<out_path<<std::endl;}  else{ std::cerr<< "Output directory previously created: "<<out_path<<std::endl;}
	out_path +=  out_dir;																													if(verbosity>0) cout <<"Creating output sub-directories: "<< out_path <<std::endl;
	boost::filesystem::create_directory(out_path);
	out_path += "/";																														if(verbosity>0) cout << "\n createFolders_chk 1\n" << flush;

	boost::filesystem::path temp_path = out_path;																							// Vector of device buffer names
																																			// imgmem[2],  gxmem[2], gymem[2], g1mem[2],  k_map_mem[2], SE3_map_mem[2], dist_map_mem[2];
	/*
		 "imgmem",						mm_size_bytes_C4,  		.HSV 3chan image. Also receives HSV_grad (2chan per image, 4 images per map) from buildDepthCostVol
		 "imgmem_blurred",				mm_size_bytes_C4,  		.HSV 3chan, before mipmap
		 "keyframe_imgmem",				2*mm_size_bytes_C4,		.HSV_grad (2chan per image, 4 images per map)

		 "gxmem",						mm_size_bytes_C4, 		.rgba 4chan
		 "gymem",						.""
		 "g1mem",						.""
		 "keyframe_g1mem",				.""

		 "SE3_grad_map_mem",			mm_size_bytes_C1*6*8	.rgba 4chan, 6 gradient maps,
		 "keyframe_SE3_grad_map_mem",	.""

		 "SE3_map_mem",					mm_size_bytes_C1*12		.Green-Blue,  rgba 4chan
		 "SE3_incr_map_mem",			mm_size_bytes_C1*24		.4chan rgba, alpha continuous scale ############## broken,
		 "SE3_rho_map_mem",				mm_size_bytes_C4,  		.4chan rgba, alpha continuous scale ############## broken,

		 "SO3_incr_map_mem",			uses SE3 buffer			.4chan rgba, alpha continuous scale ############## broken,
		 "SO3_rho_map_mem",				""						.4chan rgba, alpha continuous scale ############## broken,

		 "basemem", 					image_size_bytes,  		.3chan rgb
		 "keyframe_basemem",			## no buffer allocated?  currently not used ?

		 "depth_mem",					mm_size_bytes_C1		.1chan?		// used in RunCL::load_GT_depth, convert_depth, precomp_param_maps, transform_depthmap, & Dynamic_slam::initialize_keyframe_from_tracking.
		 "keyframe_depth_mem",			mm_size_bytes_C1		.1chan		// used by tracking
		 "key_frame_depth_map_src",		mm_size_bytes_C1	 	.""			// used in RunCL::initializeDepthCostVol( cl_mem key_frame_depth_map_src)

		 "depth_GT",					mm_size_bytes_C1,		.1chan, before mipmap
		 "dmem",						mm_size_bytes_C1,		.1chan
		 "amem", 						.""

		 "lomem",						mm_size_bytes_C1,		.1chan
		 "himem",						.""

		 "qmem",						2 * mm_size_bytes_C1, 	.A_count, Q_count
		 "qmem2",						.""

		 "cdatabuf",					mm_vol_size_bytes, 		1chan, 64layer volume
		 "cdatabuf_8chan",				mm_vol_size_bytes*8, 	1 folder per vol/8chan/ 64layers

		 "hdatabuf",					1mm_vol_size_bytes, 	chan, 64layer volume

		 "img_sum_buf",					2 * mm_vol_size_bytes,	4chan rgba, only used when debugging. Written to by RunCL::updateDepthCostVol(..)
		 "HSV_grad_mem",				2*mm_size_bytes_C4,  	2chan per image, 4 images per map,

		 "dmem_disparity"				rgb image ?
		 */
	std::vector<std::string> names = {"imgmem", "imgmem_blurred", "keyframe_imgmem", "keyframe_imgmem_HSV_grad", "gxmem", "gymem", "g1mem", "keyframe_g1mem", \
										"SE3_grad_map_mem", "keyframe_SE3_grad_map_mem", \
										"SE3_map_mem", \
										"SE3_incr_map_mem", "SE3_rho_map_mem", \
										"SO3_incr_map_mem", "SO3_rho_map_mem", \
										\
										"basemem", "keyframe_basemem", "depth_mem", "keyframe_depth_mem", \
										"key_frame_depth_map_src", "depth_GT", \
										"dmem","amem","lomem","himem","qmem","qmem2","cdatabuf","cdatabuf_8chan","hdatabuf","img_sum_buf", \
										"HSV_grad_mem", "dmem_disparity" \
	};
	std::pair<std::string, boost::filesystem::path> tempPair;

	for (std::string key : names){
		temp_path = out_path;
		temp_path += key;
		tempPair = {key, temp_path};
		paths.insert(tempPair);
		boost::filesystem::create_directory(temp_path);
		temp_path += "/png/";
		boost::filesystem::create_directory(temp_path);
	}
																																			if(verbosity>0) {
																																				cout << "\nRunCL::createFolders() chk1\n";			// print the folder paths
																																				cout << "KEY\tPATH\n";
																																				for (auto itr=paths.begin(); itr!=paths.end(); ++itr) { cout<<"First:["<< itr->first << "]\t:\t Second:"<<itr->second<<"\n"; }
																																				cout<<"\npaths.at(\"basemem\")="<<paths.at("basemem")<<"\n"<<flush;
																																			}
}

void RunCL::ReadOutput(uchar* outmat) {
		ReadOutput(outmat, amem,  (baseImage_width * baseImage_height * sizeof(float)) );
}

void RunCL::ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/) {
		cl_event readEvt;
		cl_int status;
														//cout<<"\nReadOutput: "<<flush;
														//cout<<"&outmat="<<&outmat<<", buf_mem="<<buf_mem<<", data_size="<<data_size<<", offset="<<offset<<"\n"<<flush;
		status = clEnqueueReadBuffer(dload_queue,			// command_queue
											buf_mem,		// buffer
											CL_FALSE,		// blocking_read
											offset,			// offset
											data_size,		// size
											outmat,			// pointer
											0,				// num_events_in_wait_list
											NULL,			// event_waitlist				needs to know about preceeding events:
											&readEvt);		// event
														if (status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer(..) status=" << checkerror(status) <<"\n"<<flush; exit_(status);}
															//else if(verbosity>0) cout <<"\nclEnqueueReadBuffer(..)"<<flush;

		status = clFlush(dload_queue);					if (status != CL_SUCCESS) { cout << "\nclFlush(m_queue) status = " 		<< checkerror(status) <<"\n"<<flush; exit_(status);}
															//else if(verbosity>0) cout <<"\nclFlush(..)"<<flush;

		status = clWaitForEvents(1, &readEvt); 			if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status="			<< checkerror(status) <<"\n"<<flush; exit_(status);}
															//else if(verbosity>0) cout <<"\nclWaitForEvents(..)"<<flush;

		//status = clFinish(dload_queue);					if (status != CL_SUCCESS) { cout << "\nclFinish(m_queue) status = " 		<< checkerror(status) <<"\n"<<flush; exit_(status);}
															//else if(verbosity>0) cout <<"\nclFlush(..)"<<flush;
}

void RunCL::saveCostVols(float max_range)
{																				if(verbosity>0) cout<<"\nsaveCostVols: Calling DownloadAndSaveVolume";
	stringstream ss;
	ss << "saveCostVols";
	ss << (keyFrameCount*1000 + costvol_frame_num);
	DownloadAndSaveVolume(cdatabuf, 	ss.str(), paths.at("cdatabuf"), 	mm_size_bytes_C1,  mm_Image_size, CV_32FC1,  false  , max_range);
	//DownloadAndSaveVolume(hdatabuf, 	ss.str(), paths.at("hdatabuf"), 	mm_size_bytes_C1,  baseImage_size, CV_32FC1,  false  , max_range);
	//DownloadAndSaveVolume(img_sum_buf, 	ss.str(), paths.at("img_sum_buf"), 	mm_size_bytes_C1,  baseImage_size, CV_32FC1,  false  , max_range);
																				if(verbosity>0) cout <<"\ncostvol_frame_num="<<costvol_frame_num << "\ncalcCostVol chk13_finished\n" << flush;
}


void RunCL::DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range ){
	int local_verbosity_threshold = 1;
																																			if(verbosity>0) cout<<"\n\nDownloadAndSave chk0"<<flush;
																																			if(verbosity>0) cout<<"\n\nDownloadAndSave filename = ["<<folder_tiff.filename().string()<<"] "<<flush;
																																			/*
																																			cout <<", folder="<<folder_tiff<<flush;
																																			cout <<", image_size_bytes="<<image_size_bytes<<flush;
																																			cout <<", size_mat="<<size_mat<<flush;
																																			cout <<", type_mat="<<size_mat<<"\t"<<flush;
																																			*/
		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);																				// (int rows, int cols, int type)
		ReadOutput(temp_mat.data, buffer,  image_size_bytes); 																				// NB contains elements of type_mat, (CV_32FC1 for most buffers)
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1 finished ReadOutput\n\n"<<flush;
		if (temp_mat.type() == CV_16FC1)	temp_mat.convertTo(temp_mat, CV_32FC1);															// NB conversion to FP32 req for cv::sum(..).
		cv::Scalar sum = cv::sum(temp_mat);																									// NB always returns a 4 element vector.
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.1 finished ReadOutput\n\n"<<flush;
		double minVal=1, maxVal=1;
		cv::Point minLoc={0,0}, maxLoc{0,0};
		if (temp_mat.channels()==1) { cv::minMaxLoc(temp_mat, &minVal, &maxVal, &minLoc, &maxLoc); }
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.2 finished ReadOutput\n\n"<<flush;
		string type_string = checkCVtype(type_mat);
		stringstream ss;
		stringstream png_ss;
		ss << "/" << folder_tiff.filename().string() << "_" << count <<"_sum"<<sum<<"type_"<<type_string<<"min"<<minVal<<"max"<<maxVal<<"maxRange"<<max_range;
		png_ss << "/" << folder_tiff.filename().string() << "_" << count;
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.3 finished ReadOutput\n\n"<<flush;
		boost::filesystem::path folder_png = folder_tiff;
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.4 finished ReadOutput\n\n"<<flush;
		folder_tiff += ss.str();
		folder_tiff += ".tiff";
		folder_png  += "/png/";
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.5 finished ReadOutput\n\n"<<flush;
		folder_png  += png_ss.str();
		folder_png  += ".png";
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk2 filename = ["<<ss.str()<<"]"<<flush;
		cv::Mat outMat;
		if (type_mat != CV_32FC1 && type_mat != CV_16FC1 ) {
			cout << "\n\n## Error  (type_mat != CV_32FC1 or CV_16FC1) ##\n\n" << flush;
			return;
		}																																//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk3 filename = ["<<ss.str()<<"]"<<flush;
		if (max_range == 0){ temp_mat /= maxVal;}																							// Squash/stretch & shift to 0.0-1.0 range
		else if (max_range <0.0){
			temp_mat /=(-2*max_range);
			temp_mat +=0.5;
		}else{ temp_mat /=max_range;}
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk4 filename = ["<<ss.str()<<"]"<<flush;
		if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk5 filename = ["<<ss.str()<<"]"<<flush;

		temp_mat *= 256*256;
		temp_mat.convertTo(outMat, CV_16UC1);
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk6 filename = ["<<ss.str()<<"]"<<flush;

		if(png==true) cv::imwrite(folder_png.string(), outMat );
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk7 filename = ["<<ss.str()<<"],  show="<<show<<flush;

		if(show==true) {cv::imshow( ss.str(), outMat );}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk finished filename = ["<<ss.str()<<"]"<<flush;
		return;
}

void RunCL::DownloadAndSave_2Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers ){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume() vol_layers="<<vol_layers<<", max_range="<<max_range<<", folder = ["<<folder_tiff.filename().string()<<"] "<<flush;
	if (type_mat != CV_32FC2){cout <<"Error (type_mat != CV_32FC2)"<<flush; return;}

	for (uint layer=0; layer<vol_layers; layer++  ) {

		uint offset = layer * image_size_bytes;

		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);																				// (int rows, int cols, int type)
		ReadOutput(temp_mat.data, buffer,  image_size_bytes, offset); 																		// NB contains elements of type_mat, (CV_32FC1 for most buffers)
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_Chk_1, layer="<<layer<<flush;
		cv::Mat channels[4];
		split(temp_mat, channels);																											// Split u and v (col and row) channels.
		cv::Scalar sum_u = cv::sum(channels[0]);
		cv::Scalar sum_v = cv::sum(channels[1]);
		//cv::Scalar sum_w = cv::sum(channels[2]);
		channels[2] = cv::Mat::zeros( channels[1].size(), channels[1].type() );
		channels[3] = cv::Mat::ones( channels[1].size(), channels[1].type() );

																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_Chk_2"<<flush;
		cv::Mat temp_mat_u, temp_mat_v;
		cv::Mat channels_u[4] = {channels[0],channels[0]*(-1),channels[2],channels[3] };													// Compose BGR 3channel for each of u & v.
		cv::Mat channels_v[4] = {channels[1],channels[1]*(-1),channels[2],channels[3] };													// NB Blue = +ve, , Green = -ve , Red not used.
		merge(channels_u,4,temp_mat_u);																										// Origin is top right corner of the image.
		merge(channels_v,4,temp_mat_v);
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_Chk_3"<<flush;
		double minVal_u=1, maxVal_u=1,  minVal_v=1, maxVal_v=1;
		cv::Point minLoc_u={0,0}, maxLoc_u{0,0}, minLoc_v={0,0}, maxLoc_v{0,0};
		cv::minMaxLoc(channels[0], &minVal_u, &maxVal_u, &minLoc_u, &maxLoc_u);
		cv::minMaxLoc(channels[1], &minVal_v, &maxVal_v, &minLoc_v, &maxLoc_v);
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_Chk_4"<<flush;
		string type_string = checkCVtype(type_mat);
		stringstream ss_u, ss_v;
		stringstream png_ss_u, png_ss_v;

		ss_u << "/" << folder_tiff.filename().string() << "layer_"<<layer<<"_U_" << count <<"_sum"<<sum_u<<"_type_"<<type_string<<"min"<<minVal_u<<"_max"<<maxVal_u<<"_maxRange"<<max_range;
		ss_v << "/" << folder_tiff.filename().string() << "layer_"<<layer<<"_V_" << count <<"_sum"<<sum_v<<"_type_"<<type_string<<"min"<<minVal_v<<"_max"<<maxVal_u<<"_maxRange"<<max_range;

		png_ss_u << "/" << folder_tiff.filename().string() << "layer_"<<layer<<"_U_" << count;
		png_ss_v << "/" << folder_tiff.filename().string() << "layer_"<<layer<<"_V_" << count;

		boost::filesystem::path folder_png_u = folder_tiff, folder_png_v = folder_tiff, folder_tiff_u = folder_tiff, folder_tiff_v = folder_tiff;
		folder_tiff_u += ss_u.str();
		folder_tiff_u += ".tiff";

		folder_tiff_v += ss_v.str();
		folder_tiff_v += ".tiff";

		folder_png_u  += "/png/";
		folder_png_u  += png_ss_u.str();
		folder_png_u  += ".png";

		folder_png_v  += "/png/";
		folder_png_v  += png_ss_v.str();
		folder_png_v  += ".png";
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume(), max_range="<<max_range<<",   filename = ["<<ss_u.str()<<" , "<<ss_v.str()<<"]";
		cv::Mat outMat_u, outMat_v;
		if (type_mat != CV_32FC2 && type_mat != CV_16FC2 ) {
			cout << "\n\n## Error  (type_mat != CV_32FC2 or CV_16FC2) ##\n\n" << flush;
			return;
		}
		if (max_range == 0){ temp_mat_u /= maxVal_u;  temp_mat_v /= maxVal_v; }																// Squash/stretch & shift to 0.0-1.0 range
		else if (max_range <0.0){
			temp_mat_u /=(-2*max_range);
			temp_mat_v /=(-2*max_range);
			temp_mat_u +=0.5;
			temp_mat_v +=0.5;
		}else{
			temp_mat_u /=max_range;
			temp_mat_v /=max_range;
		}
		if(tiff==true){
			cv::imwrite(folder_tiff_u.string(), temp_mat_u );
			cv::imwrite(folder_tiff_v.string(), temp_mat_v );
		}
		temp_mat_u *= 256*256 *5;																											// NB This is mosty for SE3_map_mem, which is dark due to small increment of each DoF.
		temp_mat_v *= 256*256 *5;
		temp_mat_u.convertTo(outMat_u, CV_16UC4);
		temp_mat_v.convertTo(outMat_v, CV_16UC4);
		if(png==true){
			cv::imwrite(folder_png_u.string(), outMat_u );
			cv::imwrite(folder_png_v.string(), outMat_v );
		}
		if(show){
			cv::imshow( ss_u.str(), outMat_u );
			cv::imshow( ss_v.str(), outMat_v );
		}
	}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_finished"<<flush;
}

void RunCL::DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/, bool exception_tiff /*=false*/){
	int local_verbosity_threshold = 1;
	bool old_tiff = tiff;
	if (exception_tiff == true) tiff = exception_tiff;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_0    filename = ["<<folder_tiff.filename()<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
		cv::Mat temp_mat, temp_mat2;

		if (type_mat == CV_16FC3)	{
			temp_mat2 = cv::Mat::zeros (size_mat, CV_16FC3);																				//cout << "\nReading CV_16FC3. size_mat="<< size_mat<<",   temp_mat2.total()*temp_mat2.elemSize()="<< temp_mat2.total()*temp_mat2.elemSize() << flush;
			ReadOutput(temp_mat2.data, buffer,  temp_mat2.total()*temp_mat2.elemSize(),   offset );  										// baseImage.total() * baseImage.elemSize()
																																			// void ReadOutput(   uchar* outmat,   cl_mem buf_mem,   size_t data_size,   size_t offset=0)
			temp_mat = cv::Mat::zeros (size_mat, CV_32FC3);
			temp_mat2.convertTo(temp_mat, CV_32FC3);																						// NB conversion to FP32 req for cv::sum(..).
		} else {
			temp_mat = cv::Mat::zeros (size_mat, type_mat);
			ReadOutput(temp_mat.data, buffer,  image_size_bytes,   offset);
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_1, "<<flush;
		cv::Scalar 	sum = cv::sum(temp_mat);																								// NB always returns a 4 element vector.
		string 		type_string=checkCVtype(type_mat);
		double 		minVal[3]={1,1,1}, 					maxVal[3]={0,0,0};
		cv::Point 	minLoc[3]={{0,0},{0,0},{0,0}}, 		maxLoc[3]={{0,0},{0,0},{0,0}};
		vector<cv::Mat> spl;
		split(temp_mat, spl);																												// process - extract only the correct channel
		double max = 0;
		for (int i =0; i < 3; ++i){
			cv::minMaxLoc(spl[i], &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i]);
			if (maxVal[i] > max) max = maxVal[i];
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2, "<<flush;
		stringstream ss;
		stringstream png_ss;
		ss<<"/"<<folder_tiff.filename().string()<<"_"<<count<<"_sum"<<sum<<"type_"<<type_string<<"min("<<minVal[0]<<","<<minVal[1]<<","<<minVal[2]<<")_max("<<maxVal[0]<<","<<maxVal[1]<<","<<maxVal[2]<<")";
		png_ss<< "/" << folder_tiff.filename().string() << "_" << count;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U, 256);																								// NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
			cv::waitKey(100);
		}
		boost::filesystem::path folder_png = folder_tiff;
		folder_png  += "/png/";
		folder_png  += png_ss.str();
		folder_png  += ".png";

		folder_tiff += ss.str();
		folder_tiff += ".tiff";

		if (max_range == 0){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2.1, (max_range == 0)    spl[0] /= "<<maxVal[0]<<";  spl[1] /= "<<maxVal[1]<<";  spl[2] /= "<<maxVal[2]<<";"<<flush;
			spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2];
			//spl[3] = cv::Mat::ones (size_mat, CV_32FC1);																					// set alpha=1
			cv::merge(spl, temp_mat);
		}	// Squash/stretch & shift to 0.0-1.0 range
		else if (max_range <0.0){																											if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2.2, (max_range <0.0)    squeeze and shift to 0.0-1.0 "<<flush;
			spl[0] /=(-2*max_range);  spl[1] /=(-2*max_range);  spl[2] /=(-2*max_range);
			spl[0] +=0.5;  spl[1] +=0.5;  spl[2] +=0.5;
			cv::merge(spl, temp_mat);
		}else{ 																																if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2.3, (max_range > 0)     temp_mat /=max_range;"<<flush;
			temp_mat /=max_range;
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_3, "<<flush;
		cv::Mat outMat;
		if ((type_mat == CV_32FC3) || (type_mat == CV_32FC4)){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_4, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );
			temp_mat *=256;
			temp_mat.convertTo(outMat, CV_8U);
			if (type_mat == CV_32FC4){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_5, "<<flush;
				std::vector<cv::Mat> matChannels;
				cv::split(outMat, matChannels);
				//matChannels.at(3)=255;																									// set alpha=1
				cv::merge(matChannels, outMat);
			}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_6,  folder_png.string()="<< folder_png.string() <<flush;
			if(png==true)  cv::imwrite(folder_png.string(), (outMat) );																					// Has "Grayscale 16-bit gamma integer"
		}else if (type_mat == CV_8UC3){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_7, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );
			if(png==true)  cv::imwrite(folder_png.string(),  temp_mat );
		}else if (type_mat == CV_16FC3) {																									// This demonstrates that <cv::float16_t> != <cl_half> and the read/write up/download of these types needs more debugging. NB Cannot use <cv::float16_t>  to prepare  <cl_half> data to the GPU.
			/*
			//cout << "\n Writing CV_16FC3 to .tiff & .png .\n"<< flush;
			//cout << "\n temp_mat2.at<cv::float16_t>(101,100-105) = " << temp_mat2.at<cv::float16_t>(101,100) << "," << temp_mat2.at<cv::float16_t>(101,101) << ","<< temp_mat2.at<cv::float16_t>(101,102) << ","<< temp_mat2.at<cv::float16_t>(101,103) << ","<< temp_mat2.at<cv::float16_t>(101,104) << ","<< temp_mat2.at<cv::float16_t>(101,105) << ","<< flush;
			//cout << "\n temp_mat2.at<cl_half>(101,100-105) = " << temp_mat2.at<cl_half>(101,100) << "," << temp_mat2.at<cl_half>(101,101) << ","<< temp_mat2.at<cl_half>(101,102) << ","<< temp_mat2.at<cl_half>(101,103) << ","<< temp_mat2.at<cl_half>(101,104) << ","<< temp_mat2.at<cl_half>(101,105) << ","<< flush;
			//cout << "\n temp_mat2.at<cl_half>(101,100) x,y,z,w,s0,s3 = " << temp_mat2.at<cl_half3>(101,100).x << "," << temp_mat2.at<cl_half3>(101,100).y << ","<< temp_mat2.at<cl_half3>(101,100).z << ","<< temp_mat2.at<cl_half3>(101,100).w << ","<< temp_mat2.at<cl_half3>(101,100).s0 << ","<< temp_mat2.at<cl_half3>(101,100).s3 << ","<< flush;
			*/
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_8, "<<flush;
			temp_mat2 *=256;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat2 );

			temp_mat2.convertTo(outMat, CV_8UC3);
			if(png==true) cv::imwrite(folder_png.string(), (outMat) );
		}else {cout << "\n\nError RunCL::DownloadAndSave_3Channel(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}

	tiff = old_tiff;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_9, finished "<<flush;
}

void RunCL::DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers,  bool exception_tiff /*=false*/){
	int local_verbosity_threshold = 1;
																																			if(verbosity> local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSave_3Channel_volume_chk_0   costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]"<<flush;
																																				cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<size_mat<<"\t"<<flush;
																																			}
	for (uint i=0; i<vol_layers; i++) {
		stringstream ss;	ss << count << i;
		DownloadAndSave_3Channel(buffer, ss.str(), folder, image_size_bytes, size_mat, type_mat, show, max_range, i*image_size_bytes, exception_tiff);
	}
																																			if(verbosity> local_verbosity_threshold){cout << "DownloadAndSave_3Channel_volume_chk_1  finished" << flush;}
}

void RunCL::DownloadAndSave_6Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_6Channel_Chk_0    filename = ["<<folder_tiff.filename()<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
		cv::Mat temp_mat, temp_mat2;

		if (type_mat == CV_16FC3)	{
			temp_mat2 = cv::Mat::zeros (size_mat.height, 2*size_mat.width, CV_16FC3);														//cout << "\nReading CV_16FC3. size_mat="<< size_mat<<",   temp_mat2.total()*temp_mat2.elemSize()="<< temp_mat2.total()*temp_mat2.elemSize() << flush;
			ReadOutput(temp_mat2.data, buffer,  2*temp_mat2.total()*temp_mat2.elemSize(),   offset );										// baseImage.total() * baseImage.elemSize()    // void ReadOutput(   uchar* outmat,   cl_mem buf_mem,   size_t data_size,   size_t offset=0)
			temp_mat = cv::Mat::zeros (size_mat.height, 2*size_mat.width, CV_32FC3);
			temp_mat2.convertTo(temp_mat, CV_32FC3);																						// NB conversion to FP32 req for cv::sum(..).
		} else {
			temp_mat = cv::Mat::zeros (size_mat.height, 2*size_mat.width, type_mat);
			ReadOutput(temp_mat.data, buffer,  2*image_size_bytes,   2*offset);
		}

		cv::Mat mat_u, mat_v;
		mat_u = cv::Mat::zeros (size_mat, type_mat);
		mat_v = cv::Mat::zeros (size_mat, type_mat);
		//uint data_elem_size = 4*sizeof(float);
		for (int i=0; i<mat_u.total(); i++){
			float data[8];
			for (int j=0; j<8; j++){ data[j] = temp_mat.at<float>(i*8  + j) ;}

			for (int j=0; j<4; j++){
				float alpha = ( (data[j] != 0) || (data[j+4] != 0) );
				mat_u.at<float>(i*4  + j) = data[j] ;																						// NB in buffer, alphachan carries
				mat_u.at<float>(i*4  + 3) = alpha;																							// sets alpha=0 when , else alpha=1.
				mat_v.at<float>(i*4  + j) = data[j+4] ;
				mat_v.at<float>(i*4  + 3) = alpha;
			}
		}
		//cv::imshow("mat_u", mat_u);
		//cv::imshow("mat_v", mat_v);
		//cv::waitKey(-1);
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\nmat_v alpha = ";
																																				for (int px=0; px< (mat_v.rows * mat_v.cols ) ; px += 1000){
																																					cout <<", " << mat_v.at<float>(px*4  + 3);
																																				}cout << flush;
																																			}
		SaveMat(mat_u, type_mat,  folder_tiff,  show,  max_range, "mat_u", count);
		SaveMat(mat_v, type_mat,  folder_tiff,  show,  max_range, "mat_v", count);
}

void RunCL::DownloadAndSave_HSV_grad(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_HSV_grad_Chk_0    filename = ["<<folder_tiff.filename()<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
		cv::Mat temp_mat, temp_mat2;

		if (type_mat != CV_32FC(8))	{cout <<"\nRunCL::DownloadAndSave_HSV_grad(...)  type_mat != CV_32FC(8)\n"<< flush; return;}

		temp_mat = cv::Mat::zeros (size_mat.height, size_mat.width, type_mat);
		ReadOutput(temp_mat.data, buffer,  image_size_bytes,   offset);

		cv::Mat mat_H, mat_SV, mat_Sgrad, mat_Vgrad;

		mat_H 		= cv::Mat::zeros (size_mat, CV_32FC4);
		mat_SV 		= cv::Mat::zeros (size_mat, CV_32FC3);
		mat_Sgrad 	= cv::Mat::zeros (size_mat, CV_32FC3);
		mat_Vgrad 	= cv::Mat::zeros (size_mat, CV_32FC3);

		//uint data_elem_size = 4*sizeof(float);
		for (int i=0; i<mat_H.total(); i++){
			float data[8];
			for (int j=0; j<8; j++){ data[j] = temp_mat.at<float>(i*8  + j) ;}

			mat_H.at<float>(i*4 ) 		= data[0];
			mat_H.at<float>(i*4 +1) 	= data[1];
			mat_H.at<float>(i*4 +3) 	= 1.0;			// alpha

			mat_SV.at<float>(i*3 ) 		= data[2];
			mat_SV.at<float>(i*3 +1) 	= data[3];
			//mat_SV.at<float>(i*3 +3) 	= 1.0;			// alpha

			mat_Sgrad.at<float>(i*3 ) 	= data[4];
			mat_Sgrad.at<float>(i*3 +1) = data[5];
			//mat_Sgrad.at<float>(i*3 +3) = 1.0;		// alpha

			mat_Vgrad.at<float>(i*3 ) 	= data[6];
			mat_Vgrad.at<float>(i*3 +1) = data[7];
			mat_Vgrad.at<float>(i*3 +3) = 1.0;			// alpha
		}
		bool old_tiff = tiff;
		tiff=true;
		SaveMat(mat_H,     CV_32FC4,  folder_tiff,  show,  max_range, "mat_H", count);
		tiff=old_tiff;

		SaveMat(mat_SV,    CV_32FC3,  folder_tiff,  show,  max_range, "mat_SV", count);
		SaveMat(mat_Sgrad, CV_32FC3,  folder_tiff,  show,  max_range, "mat_Sgrad", count);
		SaveMat(mat_Vgrad, CV_32FC3,  folder_tiff,  show,  max_range, "mat_Vgrad", count);
}

void RunCL::SaveMat(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_1, "<<flush;
		cv::Scalar 	sum = cv::sum(temp_mat);																								// NB always returns a 4 element vector.
		string 		type_string=checkCVtype(type_mat);
		double 		minVal[3]={1,1,1}, 					maxVal[3]={0,0,0};
		cv::Point 	minLoc[3]={{0,0},{0,0},{0,0}}, 		maxLoc[3]={{0,0},{0,0},{0,0}};
		vector<cv::Mat> spl;
		split(temp_mat, spl);																												// process - extract only the correct channel
		double max = 0;

		for (int i =0; i < 3; ++i){
																																			if(verbosity>local_verbosity_threshold)
																																				cout << "\n spl[i].empty() = "<<spl[i].empty()<<", spl[i].type() = "<< spl[i].type() << flush; //", cn = "<<cn<<", minIdx = "<< minIdx <<",  maxIdx = "<<maxIdx << flush;
			cv::minMaxLoc(spl[i], &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i]);
			if (maxVal[i] > max) max = maxVal[i];
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2, "<<flush;
		stringstream ss;
		stringstream png_ss;
		ss<<"/"<<folder_tiff.filename().string()<<"_"<<mat_name<<"_"<<count<<"__sum"<<sum<<"type_"<<type_string<<"min("<<minVal[0]<<","<<minVal[1]<<","<<minVal[2]<<")_max("<<maxVal[0]<<","<<maxVal[1]<<","<<maxVal[2]<<")";
		png_ss<< "/" << folder_tiff.filename().string() <<"_"<<mat_name<< "_" << count;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);																								// NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
		}
		boost::filesystem::path folder_png = folder_tiff;
		folder_png  += "/png";
		folder_png  += png_ss.str();
		folder_png  += ".png";

		folder_tiff += ss.str();
		folder_tiff += ".tiff";

		if (max_range == 0){ 																												if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2.1, (max_range == 0)    spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2];"<<flush;
			spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2]; 																// Squash/stretch & shift to 0.0-1.0 range
			cv::merge(spl, temp_mat);
		}
		else if (max_range <0.0){																											if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2.2, (max_range <0.0)    squeeze and shift to 0.0-1.0 "<<flush;
			spl[0] /=(-2*max_range);  spl[1] /=(-2*max_range);  spl[2] /=(-2*max_range);
			spl[0] +=0.5;  spl[1] +=0.5;  spl[2] +=0.5;
			cv::merge(spl, temp_mat);
		}else{ 																																if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2.3, (max_range > 0)     temp_mat /=max_range;"<<flush;
			temp_mat /=max_range;
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_3, "<<flush;
		cv::Mat outMat;
		if ((type_mat == CV_32FC3) || (type_mat == CV_32FC4)){																				if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_4, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );

			temp_mat.convertTo(outMat, CV_8U, 255);																							if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_6,  folder_png.string()="<< folder_png.string() <<flush;
			if(png==true)  cv::imwrite(folder_png.string(), (outMat) );																		// Has "Grayscale 16-bit gamma integer" ?
		}else if (type_mat == CV_8UC3){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_7, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );
			if(png==true)  cv::imwrite(folder_png.string(),  temp_mat );
		}																																	else {cout << "\n\nError RunCL::SaveMat(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_9, finished "<<flush;
}

void RunCL::SaveMat_1chan(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_1, "<<flush;
		cv::Scalar 	sum = cv::sum(temp_mat);																								// NB always returns a 4 element vector.
		string 		type_string=checkCVtype(type_mat);
		double 		minVal=1, 			maxVal=0;
		cv::Point 	minLoc={0,0}, 		maxLoc={0,0};
		double max = 0;
		cv::minMaxLoc(temp_mat, &minVal, &maxVal, &minLoc, &maxLoc);
		if (maxVal > max) max = maxVal;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_2, "<<flush;
		stringstream ss;
		stringstream png_ss;
		ss<<"/" << mat_name<<"__sum"<<sum<<"type_"<<type_string<<"min("<<minVal<<")_max("<<maxVal<<")";
		png_ss<< "/" << mat_name;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);																								// NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
		}
		boost::filesystem::path folder_png = folder_tiff;
		folder_png  += "/png";
		folder_png  += png_ss.str();
		folder_png  += ".png";

		folder_tiff += ss.str();
		folder_tiff += ".tiff";

		if (max_range == 0){ 																												if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_2.1, (max_range == 0)    spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2];"<<flush;
			temp_mat /= maxVal;
		}
		else if (max_range <0.0){																											if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_2.2, (max_range <0.0)    squeeze and shift to 0.0-1.0 "<<flush;
			temp_mat /=(-2*max_range);
			temp_mat +=0.5;
		}else{ 																																if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_2.3, (max_range > 0)     temp_mat /=max_range;"<<flush;
			temp_mat /=max_range;
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_3, "<<flush;
		cv::Mat outMat;
		if (type_mat == CV_32FC1){																											if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_4, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );

			temp_mat.convertTo(outMat, CV_8U, 255);																							if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_6,  folder_png.string()="<< folder_png.string() <<flush;
			if(png==true)  cv::imwrite(folder_png.string(), (outMat) );																		// Has "Grayscale 16-bit gamma integer" ?
		}																																	else {cout << "\n\nError RunCL::SaveMat_1chan(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}

																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_7, finished "<<flush;
}

void RunCL::DownloadAndSave_6Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers ){
	int local_verbosity_threshold = 1;
																																			if(verbosity> local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSave_6Channel_volume_chk_0   costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]"<<flush;
																																				cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<size_mat<<"\t"<<flush;
																																			}
	for (uint i=0; i<vol_layers; i++) {
		stringstream ss;	ss << count <<"_"<< i <<"_";
		DownloadAndSave_6Channel(buffer, ss.str(), folder, image_size_bytes, size_mat, type_mat, show, max_range, i*image_size_bytes);
	}
																																			if(verbosity> local_verbosity_threshold){cout << "DownloadAndSave_6Channel_volume_chk_1  finished" << flush;}
}

void RunCL::DownloadAndSave_8Channel(cl_mem buffer, std::string count, std::map< std::string, boost::filesystem::path > folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSave_8Channel_Chk_0    filename = ["<<folder_tiff["0"].filename()<<"] folder="<<folder_tiff["0"]<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<",  offset= "<< offset <<"\t"<<flush;
																																			}
	cv::Mat temp_mat;
	temp_mat = cv::Mat::zeros (size_mat.height, size_mat.width, CV_32FC(8) );
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_8Channel_Chk_1"
																																				<<"   temp_mat.rows = "			<< temp_mat.rows
																																				<<",  temp_mat.cols = "			<< temp_mat.cols
																																				<<",  temp_mat.elemSize() = "	<< temp_mat.elemSize()
																																				<<",  temp_mat.channels() = "	<< temp_mat.channels()
																																				<<",  size_bytes = "			<< temp_mat.total() * temp_mat.elemSize() * temp_mat.channels()
																																				<<",  8*image_size_bytes = "	<< 8*image_size_bytes
																																				<<",  mm_vol_size_bytes*8 = "	<< mm_vol_size_bytes*8 <<flush;
	ReadOutput(temp_mat.data, buffer,  image_size_bytes,   offset);

	cv::Mat mat[8];
	for (int i=0; i<8; i++){
		mat[i] = cv::Mat::zeros (size_mat, CV_32FC1);				// zero mat[i]
	}
	for (int i=0; i<mat[0].total(); i++){
		for (int j=0; j<8; j++){
			mat[j].at<float>(i) = temp_mat.at<float>(i*8  + j) ;				// copy 8 channel pixel
		}
	}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_8Channel_Chk_2";
	std::vector<std::string> names = {"0","1","2","3","4","5","6","7"};

	for (int i=0; i<8; i++){																												// for each channel
		boost::filesystem::path temp_path2 = folder_tiff[names[i]];

		stringstream ss;	ss << "channel" << names[i] << "_" << count ;
		SaveMat_1chan(mat[i], CV_32FC1,  temp_path2,  show,  max_range, ss.str(), count);
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_8Channel_Chk_3, i = "<< i <<",  temp_path2 = "<<temp_path2<<", ss.str() = "<<ss.str()<<", count = "<<count<< flush;
	}
}

void RunCL::DownloadAndSave_8Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers ){
	int local_verbosity_threshold = 1;
																																			if(verbosity> local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSave_8Channel_volume_chk_0   costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]"<<flush;
																																				cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
																																			}
	boost::filesystem::path temp_path1 = folder;
	temp_path1 += "/";
	temp_path1 += count;
	boost::filesystem::create_directory(temp_path1);																							// make new folder for this cost volume

	std::map< std::string, boost::filesystem::path > channel_paths;
	std::vector<std::string> names = {"0","1","2","3","4","5","6","7"};
	std::pair<std::string, boost::filesystem::path> tempPair;

	for (std::string key : names){
		boost::filesystem::path temp_path2 = temp_path1;
		temp_path2 += "/channel";
		temp_path2 += key;
		tempPair = {key, temp_path2};
		channel_paths.insert(tempPair);
		boost::filesystem::create_directory(temp_path2);																						// make new folders for each channel of this cost volume
		temp_path2 += "/png/";
		boost::filesystem::create_directory(temp_path2);
	}

	for (uint i=0; i<vol_layers; i++) {
		stringstream ss;	ss << count <<"__layer_"<< i <<"_";
		DownloadAndSave_8Channel(buffer, ss.str(), channel_paths, image_size_bytes, size_mat, type_mat, show, max_range, i*image_size_bytes);
																																			if(verbosity> local_verbosity_threshold){cout << "DownloadAndSave_3Channel_volume_chk_1  : ss.str() = "<< ss.str()<<",  temp_path1 = "<<temp_path1 << flush;}
	}
																																			if(verbosity> local_verbosity_threshold){cout << "DownloadAndSave_3Channel_volume_chk_2  finished " << flush;}
}

void RunCL::DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, bool exception_tiff /*=false*/){
	int local_verbosity_threshold = 0;
	bool old_tiff = tiff;
	if (exception_tiff == true) tiff = exception_tiff;
																																			if(verbosity>0) {
																																				cout<<"\n\nDownloadAndSaveVolume, costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]";
																																				cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
																																			}

	for(int i=0; i<costVolLayers; i++){
																																			if(verbosity>local_verbosity_threshold) cout << "\ncostVolLayers="<<costVolLayers<<", i="<<i<<"\t";
		size_t offset = i * image_size_bytes;
		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);																				//(int rows, int cols, int type)

		ReadOutput(temp_mat.data, buffer,  image_size_bytes, offset);
																																			if(verbosity>local_verbosity_threshold) cout << "\nRunCL::DownloadAndSaveVolume, ReadOutput completed\n"<<flush;
		if (temp_mat.type() == CV_16FC1)	temp_mat.convertTo(temp_mat, CV_32FC1);															// NB conversion to FP32 req for cv::sum(..).
		cv::Scalar sum = cv::sum(temp_mat);

		double minVal=1, maxVal=1;
		cv::Point minLoc={0,0}, maxLoc{0,0};
		if (type_mat == CV_32FC1) cv::minMaxLoc(temp_mat, &minVal, &maxVal, &minLoc, &maxLoc);

		boost::filesystem::path new_filepath = folder;
		boost::filesystem::path folder_png   = folder;

		string type_string = checkCVtype(type_mat);
		stringstream ss;
		stringstream png_ss;
		ss << "/"<< folder.filename().string() << "_" << count << "_layer"<< i <<"_sum"<<sum<<"type_"<<type_string<< "min"<<minVal<<"max"<<maxVal;
		png_ss << "/"<< folder.filename().string() << "_" << count << "_layer"<< i;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);																								// NB need CV_U8 for imshow(..)
			cv::imshow(ss.str(), temp);
		}
		new_filepath += ss.str();
		new_filepath += ".tiff";
		folder_png += "/png/";
		folder_png += png_ss.str();
		folder_png += ".png";
																																			if(verbosity>local_verbosity_threshold) cout << "\nnew_filepath.string() = "<<new_filepath.string() <<"\n";
		cv::Mat outMat;

		if (type_mat != CV_32FC1 && type_mat != CV_16FC1 ) {
			cout << "\n\n## Error  (type_mat != CV_32FC1 or CV_16FC1) ##\n\n" << flush;
			return;
		}
		if (max_range == 0){ temp_mat /= maxVal;}																							// Squash/stretch & shift to 0.0-1.0 range
		else if (max_range <0.0){
			temp_mat /=(-2*max_range);
			temp_mat +=0.5;
		}else{ temp_mat /=max_range;}

		if(tiff==true) cv::imwrite(new_filepath.string(), temp_mat );
		temp_mat *= 256*256;
		temp_mat.convertTo(outMat, CV_16UC1);
		if(png==true)  cv::imwrite(folder_png.string(), outMat );
		if(show) cv::imshow( ss.str(), outMat );
	}
	tiff = old_tiff;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////// # TODO New way: generic RunCL::DownloadAndSave_buffer(...) , and specialized calling functions.

/*
 Need to store 2D array of linear mipmap params

 When reading linearMipMap,
 1) download data
 2) copy data array into new image
    Use mipmap_params to locate data in new Mat.
 3) save png (&tiff)

 Issues
 1) data type
 2) num channels
 3) num maps in volume

 When writing CPU data to MipMap buffer
 1)

 When reading/writing linear Mipmap in kernels
 1) Minimize calculation
        read index = UID + offset
 2) Avoid smearing margins

 Make set of function aliases for each buffer.
 */
/*
//void DownloadAndSave_linear_Mipmap(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, int type_mat, bool show );

// void RunCL::DownloadAndSave_buffer (
//     cl_mem buffer,
//     std::string count,
//     //boost::filesystem::path folder_tiff,
//     std::map< std::string, boost::filesystem::path > folder_tiff,   // used in DownloadAndSave_8Channel, when called by DownloadAndSave_8Channel_volume
//
//     size_t      image_size_bytes,
//     cv::Size    size_mat,
//
//     int type_mat_out,           / *(data size and channels)* /
//     int num_channels_out,       // channels per image to write to file
//     int num_channels_in,        // channels in the buffer
//     int maps_in_vol,            // layers of depth cost vol,  //uint vol_layers,
//
//     int start_layer,            // of the mipmap
//     int stop_layer,
//
//     float max_range,            // used to scale the values in the png for visualization
//     bool exception_tiff,        // generate tiff, when tiffs are globally turned off
//     bool show                   // display the image(s).
//
//     //uint offset =0,       // passed to ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0)
//     //cv::Mat temp_mat,           // mat passed to SaveMat or SaveMat_1chan
//     //std::string mat_name        //
//
// ){
//     int type_mat_in = CV_MAKETYPE(CV_32F, num_channels_in);                                 // create temp_mat1
//     cv::Mat tempMat1(size_mat, type_mat_in);
//     float * data1 = (float*)tempMat1.data;
//     ReadOutput(tempMat1.data, buffer, image_size_bytes / *,size_t offset=0 * /);               // download buffer, to *uchar
//
//     std::vector<cv::Mat>    tempMat2_vec;                                                   // create tempMat2_vec
//     std::vector<float*>     data2;
//
//     int num_Mats = num_channels_in/num_channels_out;
//     for (int i=0; i<num_Mats; i++){
//         tempMat2_vec.push_back(cv::Mat::zeros(size_mat, CV_32FC4));
//         data2.push_back( (float*)tempMat2_vec[i].data );
//     }
//
//     for (int i=0; i<tempMat1.total(); i++){                                                 // write temp_mat1 to vector_temp_mat2
//         int a = i * num_channels_in;
//         int b = i * num_channels_out;
//         for (int j=0; j<num_Mats; j++){
//             a += j*num_channels_out;
//             for (int k=0; k<num_channels_out; k++){
//                 data2[j][b+k] = data1[a+k];
//             }
//         }
//     }
//
//     std::vector<cv::Mat>    tempMat3_vec;                                                   // create tempMat3_vec
//     std::vector<float*>     data3;
//     cv::Size size_mipmap_mat(mm_height, mm_width );
//
//     for (int i=0; i<num_Mats; i++){
//         tempMat3_vec.push_back(cv::Mat::zeros(size_mipmap_mat, CV_32FC4));
//         data3.push_back( (float*)tempMat3_vec[i].data );
//     }
//
//     for (int i=0; i<num_Mats; i++){
//         for (int j=start_layer; j<stop_layer; j++){                                             // reconstruct mipmap
//             cv:Size size_mmlayer_mat(   );
//             cv::Mat tempMat4 = cv::Mat::zeros(size_mmlayer_mat, CV_32FC4);
//
//             int offset_2    = ;  // Location of the mipmap layer in each Mat.
//             int offset_3    = ;
//             int rows        = ;
//             int cols        = ;
//             int margin      = ;
//
//             for (int k=0; k<rows; k++){
//                 for (int l=0; l<cols; l++){
//                     tempMat3_vec[i].at<Vec4f>(row,col)      = {data2[i].data[], data2[i].data[], data2[i].data[], data2[i].data[] };
//                 }
//             }
//         }
//     }
//
//     // save png(s)
//
//     // save tiff(s)
//
//     // display image
// }
*/
/*
// void RunCL::DownloadAndSave_... (){
//     DownloadAndSave_linearMipMap (
//     buffer              =,          / * cl_mem * /
//     count               =,          / * std::string * /
//     folder_tiff         =,          / * boost::filesystem::path * /
//     folder_tiff         =,          / * std::map< std::string, boost::filesystem::path > * /    // used in DownloadAndSave_8Channel, when called by DownloadAndSave_8Channel_volume
//
//     image_size_bytes    =,          / * size_t * /
//     size_mat            =,          / * cv::Size * /
//
//     type_mat            =,          / * int * /                                                 // data size and channels
//     num_channels_out    =,          / * int * /                                                 // channels per image to write to file
//     num_channels_in     =,          / * int * /                                                 // channels in the buffer
//     maps_in_vol         =,          / * int * /                                                 // layers of depth cost vol,  //uint vol_layers,
//
//     start_layer         =,          / * int * /                                                 // of the mipmap
//     stop_layer          =,          / * int * /
//
//     //uint offset / * =0 * /,                                                                   // passed to ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0)
//     max_range           =,          / * float * /                                               // used to scale the values in the png for visualization
//
//     exception_tiff      =,          / * bool * /                                                // generate tiff, when tiffs are globally turned off
//     show                =,          / * bool * /                                                // display the image(s).
//
//     temp_mat            =,          / * cv::Mat * /                                             // mat passed to SaveMat or SaveMat_1chan
//     mat_name            =           / * std::string * /
//     )
// }
*/
/*
 * The old system, all are currently 32bit float, (could be 16bit in future, would require some brand specific code)
 *
    void DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );
    depthmap_GT, keyframe_depth_mem, key_frame_depth_map_src, lomem, himem, amem, dmem, qmem, gxmem
# 1Channel: 1 channel_out, 1 channel_in, 1 map.

	void DownloadAndSave_2Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	SE3_map_mem
# 2Channel: 3or4 channels_out,

	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1, uint offset=0, bool exception_tiff=false );
	basemem, imgmem, imgmem_blurred, gxmem, gymem, g1mem, keyframe_g1mem, dmem_disparity, buffer in DownloadAndSave_3Channel_volume,
# 3Channel: 4 channels_out (inc alpha), 4 channels_in, 1 map

	void DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	SE3_incr_map_mem, SE3_rho_map_mem   ## bug in .png output.
# 3Channel_volume: 2 channels_out (+ve & -ve), 3 channels_in,

	void DownloadAndSave_6Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0);
	buffer in DownloadAndSave_6ChannelVolume(..)

	void DownloadAndSave_6Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	SE3_grad_map_mem, keyframe_SE3_grad_map_mem

	void DownloadAndSave_HSV_grad(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0 );
	HSV_grad_mem, keyframe_imgmem,

	void SaveMat(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	mat_u & mat_v                          in DownloadAndSave_6Channel(..),
	Mat_H, mat_SV, mat_Sgrad, mat_Vgrad    in DownloadAndSave_HSV_grad(..)
# HSV_grad:         2 channels_out (per image),  8 channels_in, 1 map
# 6Channel_volume:  1 channels_out (per image),  6 channels_in, 1 map


	void DownloadAndSave_8Channel(cl_mem buffer, std::string count, std::map< std::string, boost::filesystem::path > folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range , uint offset );
	buffer in DownloadAndSave_8ChannelVolume(..)

	void DownloadAndSave_8Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	cdatabuf_8chan

	void SaveMat_1chan(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	mat[i] in DownloadAndSave_8Channel(..)
# 8 channels, 1 map


	void DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, bool exception_tiff=false );
	cdatabuf, hdatabuf
# 1 channel, many maps
 *
 */
