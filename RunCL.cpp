#include "RunCL.h"

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

using namespace std;

void RunCL::testOpencl(){
	cout << "\n\nRunCL::testOpencl() ############################################################\n\n" << flush;
	cl_platform_id *platforms;
	cl_uint num_platforms;
	cl_int i, err, platform_index = -1;
	char* ext_data;
	
	size_t ext_size;
	const char icd_ext[] = "cl_khr_icd";
	/*
	cl_int clGetPlatformIDs(	cl_uint num_entries,
								cl_platform_id *platforms, 
								cl_uint *num_platforms)
	*/
																																			// Find number of platforms
	err = clGetPlatformIDs(1, NULL, &num_platforms);										if(err < 0) { perror("Couldn't find any platforms."); exit(1); }
	platforms = (cl_platform_id*)
	malloc(sizeof(cl_platform_id) * num_platforms);																							// Allocate platform array
	clGetPlatformIDs(num_platforms, platforms, NULL);																						// Initialize platform array
																																			cout << "\nnum_platforms="<<num_platforms<<"\n" << flush;
	for(i=0; i<num_platforms; i++) {
																																			cout << "\n##Platform num="<<i<<" #####################################################\n" << flush;
		char* name_data;
		/*
		cl_int clGetPlatformInfo(	cl_platform_id platform,
									cl_platform_info param_name, 
									size_t param_value_size,
									void *param_value, 
									size_t *param_value_size_ret)
		*/
																																			// Find size of name data
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &ext_size);		if(err < 0) { perror("Couldn't read platform name data."); exit(1); }
		
		name_data = (char*)malloc(ext_size);	
		clGetPlatformInfo( platforms[i],  CL_PLATFORM_NAME, ext_size, name_data, NULL);														printf("Platform %d name: %s\n", i, name_data);
		free(name_data);
																																			// Find size of extension data
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);	if(err < 0) { perror("Couldn't read extension data."); exit(1); }
		
		ext_data = (char*)malloc(ext_size);																									// Read data extension
		clGetPlatformInfo( platforms[i],  CL_PLATFORM_EXTENSIONS, ext_size, ext_data, NULL);
																																			printf("Platform %d supports extensions: %s\n", i, ext_data);
		free(ext_data);
		/*
		cl_int clGetDeviceIDs(	cl_platform_id platform,
								cl_device_type device_type, 
								cl_uint num_entries,
								cl_device_id *devices, 
								cl_uint *num_devices)
		*/
		cl_uint			num_devices		= 0;																								/*Step 2:Query the platform.*//////////////////////////////////
		cl_device_id    *devices;																											// Find number of platforms
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);		if(err < 0) { perror("Couldn't find any platforms."); continue; }
		
		devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);																// Allocate platform array
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);														// Initialize platform array
																																			cout << "\nnum_devices="<<num_devices<<"\n" << flush;
		getDeviceInfoOpencl(platforms[i]);
	}
																																			if(platform_index > -1) printf("Platform %d supports the %s extension.\n", platform_index, icd_ext);
																																			else printf("No platforms support the %s extension.\n", icd_ext);
	free(platforms);
	cout << "\nRunCL::testOpencl() finished ##################################################\n\n" << flush;
}

void RunCL::getDeviceInfoOpencl(cl_platform_id platform){
	cout << "\n#RunCL::getDeviceInfoOpencl("<< platform <<")" << "\n" << flush;
	cl_device_id *devices;
	cl_uint num_devices, addr_data;
	cl_int i, err;
	char name_data[48], ext_data[4096];
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, NULL, &num_devices); 																if(err < 0) {perror("Couldn't find any devices"); exit(1); }
	devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	for(i=0; i<num_devices; i++) {
		err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name_data), name_data, NULL); 												if(err < 0) {perror("Couldn't read extension data"); exit(1); }
		clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(ext_data), &addr_data, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(ext_data), ext_data, NULL);
																																			printf("\nDevice num: %i \nNAME: %s\nADDRESS_WIDTH: %u\nEXTENSIONS: %s \n", i, name_data, addr_data, ext_data);
	}
	free(devices);
																																			cout << "\nRunCL::getDeviceInfoOpencl("<< platform <<") finished\n" <<flush;
}

RunCL::RunCL(Json::Value obj_){
	obj = obj_;
	verbosity = obj["verbosity"].asInt();
																																			std::cout << "RunCL::RunCL verbosity = " << verbosity << std::flush;
	testOpencl();																															// Displays available OpenCL Platforms and Devices. 
																																			if(verbosity>0) cout << "\nRunCL_chk 0\n" << flush;
	createFolders( );																														/*Step1: Getting platforms and choose an available one.*/////////
	cl_uint 		numPlatforms;																											//the NO. of platforms
	cl_platform_id 	platform 		= NULL;																									//the chosen platform
	cl_int			status 			= clGetPlatformIDs(0, NULL, &numPlatforms);				if (status != CL_SUCCESS){ cout << "Error: Getting platforms!" << endl; exit_(status); };
	uint			conf_platform	= obj["opencl_platform"].asUInt();																		if(verbosity>0) cout << "numPlatforms = " << numPlatforms << ", conf_platform=" << conf_platform << "\n" << flush;
	
	if (numPlatforms > conf_platform){																										/*Choose the platform.*/
		cl_platform_id* platforms 	= (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
		
		status 	 					= clGetPlatformIDs(numPlatforms, platforms, NULL);		if (status != CL_SUCCESS){ cout << "Error: Getting platformsIDs" << endl; exit_(status); }
		
		platform 					= platforms[ conf_platform ];																			if(verbosity>0) cout << "\nplatforms[0] = "<<platforms[0]<<", \nplatforms[1] = "<<platforms[1] <<", \nplatforms[2] = "<<platforms[2] <<"\nSelected platform number :"<<conf_platform<<", cl_platform_id platform = " << platform<<"\n"<<flush;
		cl_int err;
		size_t param_value_size;
		char* platform_name;
		for (int i=0; i<numPlatforms; i++){
			err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &param_value_size);											// Find size of platform name data
			if(err < 0) { perror("Couldn't read platform name data."); exit(1); }
			platform_name = (char*)malloc(param_value_size);	
			clGetPlatformInfo( platforms[i], CL_PLATFORM_NAME, param_value_size, platform_name, NULL);										// Get platform names data
																																			cout << "\n\n platform_name = ";
																																			for(int j=0; j<5; j++){ cout << platform_name[j] ; }
																																			cout << "\n"<< flush;
			free(platform_name);
		}
		free(platforms);																	
	} else {																																cout<<"Platform num "<<conf_platform<<" not available."<<flush; exit(0);}
	
	cl_uint			numDevices		= 0;																									/*Step 2:Query the platform.*//////////////////////////////////
	cl_device_id    *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);			if (status != CL_SUCCESS) {cout << "\n3 status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	uint conf_device = obj["opencl_device"].asUInt();
	
	if (numDevices > conf_device){																											/*Choose the device*/
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status  = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}																						if (status != CL_SUCCESS) {cout << "\n4 status = " << checkerror(status) <<"\n"<<flush; exit_(status);} 
	
	cl_context_properties cps[3]={CL_CONTEXT_PLATFORM,(cl_context_properties)platform,0};													/*Step 3: Create context.*////////////////////////////////////
	m_context 	= clCreateContextFromType( cps, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);	if(status!=0) {cout<<"\n5 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	
	deviceId  	= devices[conf_device];																										/*Step 4: Create command queue & associate context.*///////////
	cl_command_queue_properties prop[] = { 0 };																								//  NB Device (GPU) queues are out-of-order execution -> need synchronization.
	m_queue 	= clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\n6 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	uload_queue = clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\n7 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	dload_queue = clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\n8 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	track_queue = clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\n9 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
																																			// Multiple queues for latency hiding: Upload, Download, Mapping, Tracking,... autocalibration, SIRFS, SPMP
																																			// NB Might want to create command queues on multiple platforms & devices.
																																			// NB might want to divde a task across multiple MPI Ranks on a multi-GPU WS or cluster.
	
	const char *filename = obj["kernel_filepath"].asCString();																				/*Step 5: Create program object*///////////////////////////////
	string sourceStr;
	status 						= convertToString(filename, sourceStr);																		if(status!=CL_SUCCESS)	{cout<<"\n10 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	const char 	*source 		= sourceStr.c_str();
	size_t 		sourceSize[] 	= { strlen(source) };
	m_program 	= clCreateProgramWithSource(m_context, 1, &source, sourceSize, NULL);
	
	status = clBuildProgram(m_program, 1, devices, NULL, NULL, NULL);																		/*Step 6: Build program.*/////////////////////
																							if (status != CL_SUCCESS){
																								printf("\nclBuildProgram failed: %d\n", status);
																								char buf[0x10000];
																								clGetProgramBuildInfo(m_program, deviceId, CL_PROGRAM_BUILD_LOG, 0x10000, buf, NULL);
																								printf("\n%s\n", buf);
																								exit_(status);
																							}
																																			/*Step 7: Create kernel objects.*//////////// 
	cvt_color_space_linear_kernel 	= clCreateKernel(m_program, "cvt_color_space_linear", 		NULL);
	img_variance_kernel				= clCreateKernel(m_program, "image_variance", 				NULL);
	reduce_kernel					= clCreateKernel(m_program, "reduce", 						NULL);
	mipmap_linear_kernel			= clCreateKernel(m_program, "mipmap_linear_flt", 			NULL);
	img_grad_kernel					= clCreateKernel(m_program, "img_grad", 					NULL);
	comp_param_maps_kernel			= clCreateKernel(m_program, "compute_param_maps", 			NULL);
	so3_grad_kernel					= clCreateKernel(m_program, "so3_grad", 					NULL);
	se3_grad_kernel					= clCreateKernel(m_program, "se3_grad", 					NULL);
	
	invert_depth_kernel				= clCreateKernel(m_program, "invert_depth", 				NULL);
	transform_depthmap_kernel		= clCreateKernel(m_program, "transform_depthmap", 			NULL);
	depth_cost_vol_kernel			= clCreateKernel(m_program, "DepthCostVol", 				NULL);
	updateQD_kernel 				= clCreateKernel(m_program, "UpdateQD", 					NULL);
	updateG_kernel  				= clCreateKernel(m_program, "UpdateG", 						NULL);
	updateA_kernel  				= clCreateKernel(m_program, "UpdateA", 						NULL);
	
	basemem=imgmem[0]=imgmem[1]=cdatabuf=hdatabuf=k2kbuf=dmem=amem=gxmem[0]=gymem[0]=g1mem[0]=gxmem[1]=gymem[1]=g1mem[1]=lomem=himem=0;		// set device pointers to zero
																																			if(verbosity>0) cout << "RunCL_constructor finished\n" << flush;
}

void RunCL::createFolders(){
																																			if(verbosity>0) cout << "\n createFolders_chk 0\n" << flush;
	std::time_t   result  = std::time(nullptr);
	std::string   out_dir = std::asctime(std::localtime(&result));
	out_dir.pop_back(); 																													// req to remove new_line from end of string.
	
	boost::filesystem::path 	out_path(boost::filesystem::current_path());
	boost::filesystem::path 	conf_outpath( obj["out_path"].asString() );
																																			cout << "\nconf_outpath = " << conf_outpath ;
	if (conf_outpath.empty()  ) {	// ||  conf_outpath.is_absolute()
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
	std::vector<std::string> names = {"imgmem[0]", "imgmem[1]", "gxmem[0]", "gxmem[1]", "gymem[0]", "gymem[1]", "g1mem[0]", "g1mem[1]",  \
										"SE3_grad_map_mem[0]", "SE3_grad_map_mem[!0]", "SE3_grad_map_mem[1]", "SE3_grad_map_mem[!1]", "keyframe_SE3_grad_map_mem", \
										"SE3_map_mem", "SE3_incr_map_mem", "SO3_incr_map_mem", "SO3_rho_map_mem", "SE3_rho_map_mem", \
										"basemem", "depth_mem", "keyframe_basemem", "keyframe_g1mem", "keyframe_imgmem", "keyframe_depth_mem", \
										"depth_GT", "dmem","amem","lomem","himem","qmem","cdatabuf","hdatabuf","img_sum_buf", \
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

void RunCL::DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range ){
	int local_verbosity_threshold = 1;
																																			//if(verbosity>0) cout<<"\n\nDownloadAndSave chk0"<<flush;
																																			if(verbosity>0) cout<<"\n\nDownloadAndSave filename = ["<<folder_tiff.filename().string()<<"] "<<flush;
																																			/*
																																			cout <<", folder="<<folder_tiff<<flush;
																																			cout <<", image_size_bytes="<<image_size_bytes<<flush;
																																			cout <<", size_mat="<<size_mat<<flush;
																																			cout <<", type_mat="<<size_mat<<"\t"<<flush;
																																			*/
		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);																				// (int rows, int cols, int type)
		ReadOutput(temp_mat.data, buffer,  image_size_bytes); 																				// NB contains elements of type_mat, (CV_32FC1 for most buffers)
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave finished ReadOutput\n\n"<<flush;
		if (temp_mat.type() == CV_16FC1)	temp_mat.convertTo(temp_mat, CV_32FC1);															// NB conversion to FP32 req for cv::sum(..).	
		cv::Scalar sum = cv::sum(temp_mat);																									// NB always returns a 4 element vector.

		double minVal=1, maxVal=1;
		cv::Point minLoc={0,0}, maxLoc{0,0};
		if (temp_mat.channels()==1) { cv::minMaxLoc(temp_mat, &minVal, &maxVal, &minLoc, &maxLoc); }
		string type_string = checkCVtype(type_mat);
		stringstream ss;
		stringstream png_ss;
		ss << "/" << folder_tiff.filename().string() << "_" << count <<"_sum"<<sum<<"type_"<<type_string<<"min"<<minVal<<"max"<<maxVal<<"maxRange"<<max_range;
		png_ss << "/" << folder_tiff.filename().string() << "_" << count;
		boost::filesystem::path folder_png = folder_tiff;
		folder_tiff += ss.str();
		folder_tiff += ".tiff";
		folder_png  += "/png/";

		folder_png  += png_ss.str();
		folder_png  += ".png";
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave filename = ["<<ss.str()<<"]";
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

		cv::imwrite(folder_tiff.string(), temp_mat );
		temp_mat *= 256*256;
		temp_mat.convertTo(outMat, CV_16UC1);
		cv::imwrite(folder_png.string(), outMat );
		if(show) cv::imshow( ss.str(), outMat );
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
		cv::imwrite(folder_tiff_u.string(), temp_mat_u );
		cv::imwrite(folder_tiff_v.string(), temp_mat_v );
		temp_mat_u *= 256*256 *5;																											// NB This is mosty for SE3_map_mem, which is dark due to small increment of each DoF.
		temp_mat_v *= 256*256 *5;
		temp_mat_u.convertTo(outMat_u, CV_16UC4);
		temp_mat_v.convertTo(outMat_v, CV_16UC4);
		cv::imwrite(folder_png_u.string(), outMat_u );
		cv::imwrite(folder_png_v.string(), outMat_v );
		
		if(show){ 
			cv::imshow( ss_u.str(), outMat_u );
			cv::imshow( ss_v.str(), outMat_v );
		}
	}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_finished"<<flush;
}

void RunCL::DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_0    filename = ["<<folder_tiff.filename()<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
		cv::Mat temp_mat, temp_mat2;
		
		if (type_mat == CV_16FC3)	{
			temp_mat2 = cv::Mat::zeros (size_mat, CV_16FC3);																				//cout << "\nReading CV_16FC3. size_mat="<< size_mat<<",   temp_mat2.total()*temp_mat2.elemSize()="<< temp_mat2.total()*temp_mat2.elemSize() << flush;
			ReadOutput(temp_mat2.data, buffer,  temp_mat2.total()*temp_mat2.elemSize(),   offset );  										// baseImage.total() * baseImage.elemSize()    // void ReadOutput(   uchar* outmat,   cl_mem buf_mem,   size_t data_size,   size_t offset=0)
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
			temp_mat.convertTo(temp, CV_8U);																								// NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
		}
		boost::filesystem::path folder_png = folder_tiff;
		folder_png  += "/png/";
		folder_png  += png_ss.str();
		folder_png  += ".png";

		folder_tiff += ss.str();
		folder_tiff += ".tiff";
		
		if (max_range == 0){ 																												if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2.1, (max_range == 0)    spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2];"<<flush;
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
			cv::imwrite(folder_tiff.string(), temp_mat );
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
			cv::imwrite(folder_png.string(), (outMat) );																					// Has "Grayscale 16-bit gamma integer"
		}else if (type_mat == CV_8UC3){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_7, "<<flush;
			cv::imwrite(folder_tiff.string(), temp_mat );
			cv::imwrite(folder_png.string(),  temp_mat );
		}else if (type_mat == CV_16FC3) {																									// This demonstrates that <cv::float16_t> != <cl_half> and the read/write up/download of these types needs more debugging. NB Cannot use <cv::float16_t>  to prepare  <cl_half> data to the GPU.
			/*
			//cout << "\n Writing CV_16FC3 to .tiff & .png .\n"<< flush;
			//cout << "\n temp_mat2.at<cv::float16_t>(101,100-105) = " << temp_mat2.at<cv::float16_t>(101,100) << "," << temp_mat2.at<cv::float16_t>(101,101) << ","<< temp_mat2.at<cv::float16_t>(101,102) << ","<< temp_mat2.at<cv::float16_t>(101,103) << ","<< temp_mat2.at<cv::float16_t>(101,104) << ","<< temp_mat2.at<cv::float16_t>(101,105) << ","<< flush; 
			//cout << "\n temp_mat2.at<cl_half>(101,100-105) = " << temp_mat2.at<cl_half>(101,100) << "," << temp_mat2.at<cl_half>(101,101) << ","<< temp_mat2.at<cl_half>(101,102) << ","<< temp_mat2.at<cl_half>(101,103) << ","<< temp_mat2.at<cl_half>(101,104) << ","<< temp_mat2.at<cl_half>(101,105) << ","<< flush; 
			//cout << "\n temp_mat2.at<cl_half>(101,100) x,y,z,w,s0,s3 = " << temp_mat2.at<cl_half3>(101,100).x << "," << temp_mat2.at<cl_half3>(101,100).y << ","<< temp_mat2.at<cl_half3>(101,100).z << ","<< temp_mat2.at<cl_half3>(101,100).w << ","<< temp_mat2.at<cl_half3>(101,100).s0 << ","<< temp_mat2.at<cl_half3>(101,100).s3 << ","<< flush; 
			*/
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_8, "<<flush;
			temp_mat2 *=256;
			cv::imwrite(folder_tiff.string(), temp_mat2 );
			
			temp_mat2.convertTo(outMat, CV_8UC3);
			cv::imwrite(folder_png.string(), (outMat) );
		}else {cout << "\n\nError RunCL::DownloadAndSave_3Channel(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_9, finished "<<flush;
}

void RunCL::DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers ){
	int local_verbosity_threshold = 1;
																																			if(verbosity> local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSave_3Channel_volume_chk_0   costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]"<<flush;
																																				cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<size_mat<<"\t"<<flush;
																																			}
	for (uint i=0; i<vol_layers; i++) {
		stringstream ss;	ss << count << i;
		DownloadAndSave_3Channel(buffer, ss.str(), folder, image_size_bytes, size_mat, type_mat, show, max_range, i*image_size_bytes);
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
		// 
		
		cv::Mat mat_u, mat_v;
		mat_u = cv::Mat::zeros (size_mat, type_mat);
		mat_v = cv::Mat::zeros (size_mat, type_mat);
		//uint data_elem_size = 4*sizeof(float);
		for (int i=0; i<mat_u.total(); i++){
			float data[8];
			for (int j=0; j<8; j++){ data[j] = temp_mat.at<float>(i*8  + j) ;}
			for (int j=0; j<4; j++){
				mat_u.at<float>(i*4  + j) = data[j] ;
				mat_v.at<float>(i*4  + j) = data[j+4] ;																						// NB in buffer, alphachan carries 
				mat_u.at<float>(i*4  + 3) = (data[j] != 0); // 1.0f;																		// sets alpha=0 when , else alpha=1.
				mat_v.at<float>(i*4  + 3) = (data[j] != 0); // 1.0f;
			}
		}
		//cv::imshow("mat_u", mat_u);
		//cv::imshow("mat_v", mat_v);
		cv::waitKey(-1);
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\nmat_v alpha = ";
																																				for (int px=0; px< (mat_v.rows * mat_v.cols ) ; px += 1000){
																																					cout <<", " << mat_v.at<float>(px*4  + 3);
																																				}cout << flush;
																																			}
		SaveMat(mat_u, type_mat,  folder_tiff,  show,  max_range, "mat_u", count);
		SaveMat(mat_v, type_mat,  folder_tiff,  show,  max_range, "mat_v", count);
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
			cv::minMaxLoc(spl[i], &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i]);
			if (maxVal[i] > max) max = maxVal[i];
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2, "<<flush;
		stringstream ss;
		stringstream png_ss;
		ss<<"/"<<folder_tiff.filename().string()<<"_"<<mat_name<<"_"<<count<<"_sum"<<sum<<"type_"<<type_string<<"min("<<minVal[0]<<","<<minVal[1]<<","<<minVal[2]<<")_max("<<maxVal[0]<<","<<maxVal[1]<<","<<maxVal[2]<<")";
		png_ss<< "/" << folder_tiff.filename().string() <<"_"<<mat_name<< "_" << count;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);																								// NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
		}
		boost::filesystem::path folder_png = folder_tiff;
		folder_png  += "/png/";
		folder_png  += png_ss.str();
		folder_png  += ".png";

		folder_tiff += ss.str();
		folder_tiff += ".tiff";
		
		if (max_range == 0){ 																												if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2.1, (max_range == 0)    spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2];"<<flush;
			spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2]; 
			//spl[3] = cv::Mat::ones (size_mat, CV_32FC1);																					// set alpha=1
			cv::merge(spl, temp_mat);
		}	// Squash/stretch & shift to 0.0-1.0 range
		else if (max_range <0.0){																											if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2.2, (max_range <0.0)    squeeze and shift to 0.0-1.0 "<<flush;
			spl[0] /=(-2*max_range);  spl[1] /=(-2*max_range);  spl[2] /=(-2*max_range); 
			spl[0] +=0.5;  spl[1] +=0.5;  spl[2] +=0.5;
			cv::merge(spl, temp_mat);
		}else{ 																																if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2.3, (max_range > 0)     temp_mat /=max_range;"<<flush;
			temp_mat /=max_range;
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_3, "<<flush;
		cv::Mat outMat;
		if ((type_mat == CV_32FC3) || (type_mat == CV_32FC4)){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_4, "<<flush;
			cv::imwrite(folder_tiff.string(), temp_mat );
			temp_mat *=256;
			temp_mat.convertTo(outMat, CV_8U);
			if (type_mat == CV_32FC4){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_5, "<<flush;
				std::vector<cv::Mat> matChannels;
				cv::split(outMat, matChannels);
				//matChannels.at(3)=255;																									// set alpha=1
				cv::merge(matChannels, outMat);
			}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_6,  folder_png.string()="<< folder_png.string() <<flush;
			cv::imwrite(folder_png.string(), (outMat) );																					// Has "Grayscale 16-bit gamma integer"
		}else if (type_mat == CV_8UC3){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_7, "<<flush;
			cv::imwrite(folder_tiff.string(), temp_mat );
			cv::imwrite(folder_png.string(),  temp_mat );
		}																																	else {cout << "\n\nError RunCL::SaveMat(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_9, finished "<<flush;
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
																																			if(verbosity> local_verbosity_threshold){cout << "DownloadAndSave_3Channel_volume_chk_1  finished" << flush;}
}

void RunCL::DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range ){
	int local_verbosity_threshold = 1;
																																			if(verbosity>0) {
																																				cout<<"\n\nDownloadAndSaveVolume, costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]";
																																				cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<size_mat<<"\t"<<flush;
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

		cv::imwrite(new_filepath.string(), temp_mat );
		temp_mat *= 256*256;
		temp_mat.convertTo(outMat, CV_16UC1);
		cv::imwrite(folder_png.string(), outMat );
		if(show) cv::imshow( ss.str(), outMat );
	}
}

void RunCL::computeSigmas(float epsilon, float theta, float L, float &sigma_d, float &sigma_q ){
		float mu	= 2.0*std::sqrt((1.0/theta)*epsilon) /L;
		sigma_d		=  mu / (2.0/ theta)  ;
		sigma_q 	=  mu / (2.0*epsilon) ;
}


void RunCL::initialize_fp32_params(){
	fp32_params[MAX_INV_DEPTH]	=  1/obj["min_depth"].asFloat()		;																		// This works: Initialize 'params[]' from conf.json . 
	fp32_params[MIN_INV_DEPTH]	=  1/obj["max_depth"].asFloat()		;
				//INV_DEPTH_STEP	;
	fp32_params[ALPHA_G]		=    obj["alpha_g"].asFloat()		;
	fp32_params[BETA_G]			=    obj["beta_g"].asFloat()		;
	fp32_params[EPSILON]		=    obj["epsilon"].asFloat()		;
				//SIGMA_Q ;
				//SIGMA_D ;
	fp32_params[THETA]			=    obj["thetaStart"].asFloat()	;
	fp32_params[LAMBDA]			=    obj["lambda"].asFloat()		;
	fp32_params[SCALE_EAUX]		=    obj["scale_E_aux"].asFloat()	;
	fp32_params[SE3_LM_A]		=    obj["SE3_LM_A"].asFloat()		;
	fp32_params[SE3_LM_B]		=    obj["SE3_LM_B"].asFloat()		;
}


void RunCL::initialize(){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_chk0\n\n" << flush;
																																			if(baseImage.empty()){cout <<"\nError RunCL::initialize() : runcl.baseImage.empty()"<<flush; exit(0); }
	image_size_bytes	= baseImage.total() * baseImage.elemSize();																			// Constant parameters of the base image
	image_size_bytes_C1	= baseImage.total() * sizeof(float);
	costVolLayers 		= 2*( 1 + obj["layers"].asUInt() );
	baseImage_size 		= baseImage.size();
	baseImage_type 		= baseImage.type();
	baseImage_width		= baseImage.cols;
	baseImage_height	= baseImage.rows;
	layerstep 			= baseImage_width * baseImage_height;
	
	mm_num_reductions	= obj["num_reductions"].asUInt();																					// Constant parameters of the mipmap, (as opposed to per-layer mipmap_buf)
	mm_start			= 0;
	mm_stop				= mm_num_reductions;
	mm_gaussian_size	= obj["gaussian_size"].asUInt();
	mm_margin			= obj["MipMap_margin"].asUInt() * mm_num_reductions;
	mm_width 			= baseImage_width  + 2 * mm_margin;
	mm_height 			= baseImage_height * 2.1  + 2 * mm_margin;  // 1.5
	mm_layerstep		= mm_width * mm_height;
	
	cv::Mat temp(mm_height, mm_width, CV_32FC3);
	mm_Image_size		= temp.size();
	mm_Image_type		= temp.type();
	mm_size_bytes_C3	= temp.total() * temp.elemSize() ;																					// for mipmaps with CV_16FC3  mm_width*mm_height*fp16_size;// for FP16 'half', or BF16 on Tensor cores
	mm_size_bytes_C4	= temp.total() * 4 * sizeof(float);
	cv::Mat temp2(mm_height, mm_width, CV_32FC1);
	mm_size_bytes_C1	= temp2.total() * temp2.elemSize();
	mm_vol_size_bytes	= mm_size_bytes_C1 * costVolLayers;
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_chk1\n\n" << flush;
																																			// Get the maximum work group size for executing the kernel on the device ///////
																																			// From https://github.com/rsnemmen/OpenCL-examples/blob/e2c34f1dfefbd265cfb607c2dd6c82c799eb322a/square_array/square.c
	cl_int 				status;
	status = clGetKernelWorkGroupInfo(cvt_color_space_linear_kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size), &local_work_size, NULL); 	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
																																			// Number of total work items, calculated here after 1st image is loaded &=> know the size.
																																			// NB localSize must be devisor
																																			// NB global_work_size must be a whole number of "Preferred work group size multiple" for Nvidia. 
																																			// i.e. global_work_size should be slightly more than the number of point or pixels to be processed.
	global_work_size 	= ceil( (float)layerstep/(float)local_work_size ) * local_work_size;
	mm_global_work_size = ceil( (float)mm_layerstep/(float)local_work_size ) * local_work_size;
																																			if(verbosity>local_verbosity_threshold){ 
																																				cout << "\n\nRunCL::initialize_chk1.2,\t global_work_size="<<global_work_size<<",\t mm_global_work_size="<<mm_global_work_size <<"\n" << flush;
																																				cout << "layerstep="<<layerstep <<",\t mm_layerstep"<< mm_layerstep<<"\n\n" << flush;
																																				cout<<"\nglobal_work_size="<<global_work_size<<", local_work_size="<<local_work_size<<", deviceId="<<deviceId<<"\n"<<flush;
																																				cout<<"\nlayerstep=mm_width*mm_height="<<mm_width<<"*"<<mm_height<<"="<<layerstep<<",\tsizeof(layerstep)="<< sizeof(layerstep) <<",\tsizeof(int)="<< sizeof(int) <<flush;
																																				cout<<"\n";
																																				cout<<"\nallocatemem chk1, baseImage.total()=" << baseImage.total() << ", sizeof(float)="<< sizeof(float)<<flush;
																																				cout<<"\nbaseImage.elemSize()="<< baseImage.elemSize()<<", baseImage.elemSize1()="<<baseImage.elemSize1()<<flush;
																																				cout<<"\nbaseImage.type()="<< baseImage.type() <<", sizeof(baseImage.type())="<< sizeof(baseImage.type())<<flush;
																																				cout<<"\n";
																																				cout<<"\nallocatemem chk2, image_size_bytes="<< image_size_bytes <<  ", sizeof(float)="<< sizeof(float)<<flush;
																																				cout<<"\n";
																																				cout<<"\n"<<", fp16_size ="<< fp16_size   <<", mm_margin="     << mm_margin       <<", mm_width ="     <<  mm_width       <<flush;
																																				cout<<"\n"<<", mm_height ="<< mm_height   <<", mm_Image_size ="<<  mm_Image_size  <<", mm_Image_type ="<< mm_Image_type   <<flush;
																																				cout<<"\n"<<", mm_size_bytes_C1="<< mm_size_bytes_C1  <<", mm_size_bytes_C3="<< mm_size_bytes_C3 <<", mm_size_bytes_C4="<< mm_size_bytes_C4 <<", mm_vol_size_bytes ="<<  mm_vol_size_bytes  <<flush;
																																				cout<<"\n";
																																				cout<<"\n"<<", temp.elemSize() ="<< temp.elemSize()   <<", temp2.elemSize()="<< temp2.elemSize() <<flush;
																																				cout<<"\n"<<", temp.total() ="<< temp.total()         <<", temp2.total()="   << temp2.total()    <<flush;
																																			}
	initialize_fp32_params();
																																			if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::initialize_chk3.8\n\n" << flush;
	uint_params[PIXELS]			= 	baseImage_height * baseImage_width ;
	uint_params[ROWS]			= 	baseImage_height ;
	uint_params[COLS]			= 	baseImage_width ;
	uint_params[LAYERS]			= 	obj["layers"].asUInt() ;
	uint_params[MARGIN]			= 	mm_margin ;
	uint_params[MM_PIXELS]		= 	mm_height * mm_width ;
	uint_params[MM_ROWS]		= 	mm_height ;
	uint_params[MM_COLS]		= 	mm_width ;
																																			if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::initialize_chk3.9\n\n" << flush;
	computeSigmas( obj["epsilon"].asFloat(), obj["thetaStart"].asFloat(), obj["L"].asFloat(), fp32_params[SIGMA_Q], fp32_params[SIGMA_D] );
																																			//computeSigmas( obj["epsilon"].asFloat(), obj["thetaStart"].asFloat(), obj["L"].asFloat(), cl_half_params[SIGMA_Q], cl_half_params[SIGMA_D] );
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n\nChecking fp32_params[]";
																																				cout << "\nfp32_params[0 MAX_INV_DEPTH]="	<<fp32_params[MAX_INV_DEPTH]		<<"\t\t1/obj[\"min_depth\"].asFloat()="	<<1/obj["min_depth"].asFloat();
																																				cout << "\nfp32_params[1 MIN_INV_DEPTH]="	<<fp32_params[MIN_INV_DEPTH]		<<"\t\t1/obj[\"max_depth\"].asFloat()="	<<1/obj["max_depth"].asFloat();
																																				cout << "\nfp32_params[2 INV_DEPTH_STEP]="	<<fp32_params[INV_DEPTH_STEP];
																																				cout << "\nfp32_params[3 ALPHA_G]="			<<fp32_params[ALPHA_G]				<<"\t\tobj[\"alpha_g\"].asFloat()="		<<obj["alpha_g"].asFloat();
																																				cout << "\nfp32_params[4 BETA_G]="			<<fp32_params[BETA_G]				<<"\t\tobj[\"beta_g\"].asFloat()="		<<obj["beta_g"].asFloat();
																																				cout << "\nfp32_params[5 EPSILON]="			<<fp32_params[EPSILON]				<<"\t\tobj[\"epsilon\"].asFloat()="		<<obj["epsilon"].asFloat();
																																				cout << "\nfp32_params[6 SIGMA_Q]="			<<fp32_params[SIGMA_Q];
																																				cout << "\nfp32_params[7 SIGMA_D ]="		<<fp32_params[SIGMA_D ];
																																				cout << "\nfp32_params[8 THETA]="			<<fp32_params[THETA]				<<"\t\tobj[\"thetaStart\"].asFloat()="	<<obj["thetaStart"].asFloat();
																																				cout << "\nfp32_params[9 LAMBDA]="			<<fp32_params[LAMBDA]				<<"\t\tobj[\"lambda\"].asFloat()="		<<obj["lambda"].asFloat();
																																				cout << "\nfp32_params[10 SCALE_EAUX]="		<<fp32_params[SCALE_EAUX]			<<"\t\tobj[\"scale_E_aux\"].asFloat()="	<<obj["scale_E_aux"].asFloat();
																																				cout << "\n" << flush;
																																			}
	
	for (int i=0; i<3; i++){ fp32_so3_k2k[i+ i*3]=1.0; }																					// initialize fp32_so3_k2k & fp32_k2k as 'unity' transform, i.e. zero rotation & zero translation.	
	for (int i=0; i<4; i++){ fp32_k2k[i+ i*4]    =1.0; }																					// NB instantiated as {{0}}.
	/*
	fp32_k2k[0]  =  1.0 ;	// (1,0,0,0)																								
	fp32_k2k[5]  =  1.0 ;	// (0,1,0,0)
	fp32_k2k[10] =  1.0 ;	// (0,0,1,0)
	fp32_k2k[15] =  1.0 ;	// (0,0,0,1)
	*/
																																			if(verbosity>local_verbosity_threshold) {
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
	// ####################################################################################################################################################################
																																			if(verbosity>local_verbosity_threshold) { 
																																				cout <<"	"<<endl;
																																				cout <<"	#define MiM_PIXELS			0	// for mipmap_buf, 				when launching one kernel per layer. 	Updated for each layer."<<endl;
																																				cout <<"	#define MiM_READ_OFFSET		1	// for ths layer, 				start of image data"<<endl;
																																				cout <<"	#define MiM_WRITE_OFFSET	2"<<endl;
																																				cout <<"	#define MiM_READ_COLS		3	// cols without margins"<<endl;
																																				cout <<"	#define MiM_WRITE_COLS		4"<<endl;
																																				cout <<"	#define MiM_GAUSSIAN_SIZE	5	// filter box size"<<endl;
																																				cout <<"	#define MiM_READ_ROWS		6	// rows without margins"<<endl;
																																				cout <<"	#define MiM_WRITE_ROWS		7"<<endl;
																																				cout <<"	"<<endl;
																																			}
	
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
	
	for(int reduction = 0; reduction <= mm_num_reductions+1; reduction++) {   
		num_threads[reduction]		= ceil( (float)(mipmap[MiM_PIXELS])/(float)local_work_size ) * local_work_size ;						// global_work_size formula for num_treads req for this layer. 
		for (int i=0; i<8; i++) 	{																										// Initialize the global MipMap[8*8] array. 
			MipMap[reduction*8 +i] = mipmap[i];
																																			if(verbosity>local_verbosity_threshold) { cout << "\nMipMap["<<reduction<<"*8 +"<<i<<"]="<<MipMap[reduction*8 +i] ;}
		}																																	if(verbosity>local_verbosity_threshold) { cout << endl << flush; }
		mipmap[MiM_READ_OFFSET] 	= mipmap[MiM_WRITE_OFFSET];
		mipmap[MiM_WRITE_OFFSET] 	= mipmap[MiM_WRITE_OFFSET] + read_cols_with_margin * (margin + write_rows);
		mipmap[MiM_READ_ROWS] 		= write_rows;
		write_rows					= write_rows/2;
		mipmap[MiM_READ_COLS] 		= mipmap[MiM_WRITE_COLS];
		mipmap[MiM_WRITE_COLS] 		= mipmap[MiM_WRITE_COLS]/2;
		mipmap[MiM_PIXELS]			= mipmap[MiM_READ_COLS] * mipmap[MiM_READ_ROWS];
	}
																																			// Summation buffer sizes
	se3_sum_size 		= 1 + ceil( (float)(MipMap[(mm_num_reductions+1)*8 + MiM_READ_OFFSET]) / (float)local_work_size ) ;					// i.e. num workgroups used = MiM_READ_OFFSET for 1 layer more than used / local_work_size,   will give one row of vector per group.
	uint num_DoFs		= 6 ; 																												// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
	se3_sum_size_bytes	= se3_sum_size * sizeof(float) * 4 * num_DoFs;																		if(verbosity>local_verbosity_threshold) cout <<"\n\n se3_sum_size="<< se3_sum_size<<",    se3_sum_size_bytes="<<se3_sum_size_bytes<<flush;
	se3_sum2_size_bytes = 2 * mm_num_reductions * sizeof(float) * 4 * num_DoFs;																// NB the data returned is 6xfloat4 per group, holding one float4 per 6DoF of SE3, where alpha channel=pixel count.
	
	so3_sum_size_bytes	= se3_sum_size_bytes / 2;
	so3_sum_size		= se3_sum_size ;
	//pix_sum_size		= 1 + ceil( (float)(baseImage_width * baseImage_height) / (float)local_work_size ) ;  								// i.e. num workgroups used = baseImage_width * baseImage_height / local_work_size,   will give one row of vector per group.
	pix_sum_size		= se3_sum_size;
	pix_sum_size_bytes	= pix_sum_size * sizeof(float) * 4;																					// NB the data returned is one float4 per group, for the base image, holding hsv channels plus entry[3]=pixel count.
	
}

void RunCL::mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap_call_kernel(cl_kernel "<<kernel_to_call<<", cl_command_queue "<<queue_to_call<<", "<<start<<", "<<stop<<" )_chk0"<<flush;
																																			}
	cl_event						ev;
	cl_int							res, status;
	for(int reduction = 0; reduction <= mm_num_reductions+1; reduction++) { 
																																			if(verbosity>local_verbosity_threshold) { cout<<"\n\nRunCL::mipmap_call_kernel(..) reduction="<<reduction<<"                chk1"<<flush; }
		if (reduction>=start && reduction<stop){																							// compute num threads to launch & num_pixels in reduction
																																			if(verbosity>local_verbosity_threshold) { cout<<"\nRunCL::mipmap_call_kernel(..) if (reduction>=start && reduction<stop)  chk2"<<flush; }
			res 	= clSetKernelArg(kernel_to_call, 0, sizeof(int), &reduction);							if(res!=CL_SUCCESS)			{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;
			res 	= clEnqueueNDRangeKernel(queue_to_call, kernel_to_call, 1, 0, &num_threads[reduction], &local_work_size, 0, NULL, &ev); // run mipmap_linear_kernel, NB wait for own previous iteration.
																											if (res != CL_SUCCESS)		{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
			status 	= clFlush(queue_to_call);																if (status != CL_SUCCESS)	{ cout << "\nclFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
			status 	= clWaitForEvents (1, &ev);																if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
		} 																																	// TODO execute layers in asynchronous parallel. i.e. relax clWaitForEvents.
	}
}

/*
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
*/

/*
		cl_int clEnqueueNDRangeKernel(
										cl_command_queue command_queue,
										cl_kernel kernel,
										cl_uint work_dim,
										const size_t* global_work_offset,
										const size_t* global_work_size,
										const size_t* local_work_size,
										cl_uint num_events_in_wait_list,
										const cl_event* event_wait_list,
										cl_event* event);
*/

/*
void RunCL::mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop){
	int local_verbosity_threshold = 0;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap_call_kernel(cl_kernel "<<kernel_to_call<<", cl_command_queue "<<queue_to_call<<", "<<start<<", "<<stop<<" )_chk0"<<flush;
																																			}
	cl_event						writeEvt, ev;
	cl_int							res, status;
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
	for(int reduction = 0; reduction <= mm_num_reductions+1; reduction++) {                                                                 if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2"<<flush;
																																				cout << "\nreduction="<< reduction << " , read_rows=" << mipmap[MiM_READ_ROWS]   << " ,  write_rows=" <<  write_rows  << " ,  read_cols_with_margin=" << 	read_cols_with_margin  << " ,  read_rows_with_margin=" <<  read_rows_with_margin  << " ,  margin=" << 	margin  << " ,   mipmap[MiM_READ_OFFSET]=" <<  mipmap[MiM_READ_OFFSET]  << " ,  mipmap[MiM_WRITE_OFFSET]=" <<  mipmap[MiM_WRITE_OFFSET]	  << " ,  mipmap[MiM_READ_COLS]=" <<   mipmap[MiM_READ_COLS]  << " ,   mipmap[MiM_WRITE_COLS]=" <<    mipmap[MiM_WRITE_COLS]  << " ,   mipmap[MiM_GAUSSIAN_SIZE]=" <<    mipmap[MiM_GAUSSIAN_SIZE] << endl << flush; 
                                                                                                                                            }
		if (reduction>=start && reduction<stop){																																			// compute num threads to launch & num_pixels in reduction
			size_t num_threads		= ceil( (float)(mipmap[MiM_PIXELS])/(float)local_work_size ) * local_work_size ;						// global_work_size formula  
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.1   num_threads="<<num_threads <<",  mipmap[MiM_PIXELS]="<<mipmap[MiM_PIXELS]<<flush;}
			status 	= clEnqueueWriteBuffer(uload_queue, mipmap_buf, 	CL_FALSE, 0, 8 * sizeof(uint), 	mipmap, 0, NULL, &writeEvt);		// write mipmap_buf
                                                                                                            if (status != CL_SUCCESS)	{ cout<<"\nstatus = "<<checkerror(status)<<"\n"<<flush; cout << "Error: RunCL::mipmap_call_kernel, clEnqueueWriteBuffer, mipmap_buf \n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
			res 	= clSetKernelArg(kernel_to_call, 0, sizeof(cl_mem), &mipmap_buf);						if(res!=CL_SUCCESS)			{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global uint*	mipmap_params	//3
			status 	= clFlush(m_queue); 																	if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}	// clEnqueueNDRangeKernel
			status 	= clFinish(m_queue); 																	if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
		
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.2 , num_threads="<<num_threads<<", local_work_size="<<local_work_size<<flush;}
			res = clEnqueueNDRangeKernel(queue_to_call, kernel_to_call, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev); 																// run mipmap_linear_kernel, NB wait for own previous iteration.
																											if (res != CL_SUCCESS)		{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		status = clFlush(queue_to_call);																	if (status != CL_SUCCESS)	{ cout << "\nclFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clWaitForEvents (1, &ev);																	if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}		
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.3 "<<flush;}
		}																																	if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.4 "<<flush;}
		mipmap[MiM_READ_OFFSET] 	= mipmap[MiM_WRITE_OFFSET];
		mipmap[MiM_WRITE_OFFSET] 	= mipmap[MiM_WRITE_OFFSET] + read_cols_with_margin * (margin + write_rows);
		mipmap[MiM_READ_ROWS] 		= write_rows;
		write_rows					= write_rows/2;
		mipmap[MiM_READ_COLS] 		= mipmap[MiM_WRITE_COLS];
		mipmap[MiM_WRITE_COLS] 		= mipmap[MiM_WRITE_COLS]/2;
		mipmap[MiM_PIXELS]			= mipmap[MiM_READ_COLS] * mipmap[MiM_READ_ROWS];
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk2.6 Finished one loop"<<flush;}
	}																																		if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_call_kernel(..)_chk3 Finished "<<flush;}
}
*/


void RunCL::allocatemem(){
	int local_verbosity_threshold = 1;
	stringstream 		ss;
	ss << "allocatemem";
	cl_int 				status;
	cl_event 			writeEvt;
	cl_int res;
	//imgmem[2],  gxmem[2], gymem[2], g1mem[2],  k_map_mem[2], SE3_map_mem[2], dist_map_mem[2]; // alernate copies for consecutive frames, used in SE3 tracking.
	for (int i=0; i<2; i++){
		imgmem[i]			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, mm_size_bytes_C4,  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		gxmem[i]			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C4, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		gymem[i]			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C4, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		g1mem[i]			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C4, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		k_map_mem[i]		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1*10,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // camera intrinsic map
		dist_map_mem[i]		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1*28,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // distorsion map
		SE3_grad_map_mem[i] = clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1*6*8,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // SE3_map * img grad, 6DoF*3channels=18, but 4*6=24 because hsv img gradient is held in float4
	}
	keyframe_SE3_grad_map_mem = clCreateBuffer(m_context, CL_MEM_READ_WRITE 				, mm_size_bytes_C1*6*8,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	SE3_incr_map_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1*24,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // For debugging before summation.
	SE3_map_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1*12,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	// (row, col) increment fo each parameter.
	basemem				= clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, image_size_bytes,  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // Original image CV_8UC3
	depth_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, image_size_bytes_C1,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // Copy used by tracing & auto-calib
	depth_mem_GT		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, image_size_bytes_C1,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	keyframe_basemem	= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, mm_size_bytes_C4,  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	// Depth mapping buffers
	//keyframe_gxmem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C4, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	//keyframe_gymem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C4, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	keyframe_g1mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C4, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	dmem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	amem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	lomem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	himem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	qmem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE						, 2 * mm_size_bytes_C1, 0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	cdatabuf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_vol_size_bytes, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	hdatabuf 			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_vol_size_bytes, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	img_sum_buf 		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, 2 * mm_vol_size_bytes,0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	// float debug buffer.
	fp32_param_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 16 * sizeof(float),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	k2kbuf				= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 16 * sizeof(float),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	SO3_k2kbuf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 9*sizeof(float),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // NB used in place of k2kbuf for RunCL::estimateSO3(..)
	SE3_k2kbuf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 6*16*sizeof(float),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	uint_param_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 8 * sizeof(uint),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	mipmap_buf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 8*8*sizeof(uint),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gaussian_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 9 * sizeof(float),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//  TODO load gaussian kernel & size from conf.json .
	
	se3_sum_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, se3_sum_size_bytes,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	se3_sum2_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, se3_sum2_size_bytes,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	SE3_rho_map_mem		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, mm_size_bytes_C4,  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	se3_sum_rho_sq_mem	= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, pix_sum_size_bytes,  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	img_stats_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, img_stats_size_bytes,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	pix_sum_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, pix_sum_size_bytes,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	var_sum_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, pix_sum_size_bytes,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	//reduce_param_buf	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, 8 * sizeof(uint)	,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nRunCL::allocatemem_chk3\n\n" << flush;
																																			cout << ",dmem = " 			<< dmem << endl;
																																			cout << ",amem = " 			<< amem << endl;
																																			cout << ",gxmem = " 		<< gxmem << endl;
																																			cout << ",gymem = " 		<< gymem << endl;
																																			cout << ",qmem = " 			<< qmem << endl;
																																			cout << ",g1mem = " 		<< g1mem << endl;
																																			cout << ",lomem = " 		<< lomem << endl;
																																			cout << ",himem = " 		<< himem << endl;
																																			cout << ",cdatabuf = " 		<< cdatabuf << endl;
																																			cout << ",hdatabuf = " 		<< hdatabuf << endl;
																																			cout << ",imgmem = " 		<< imgmem << endl;
																																			cout << ",basemem = " 		<< basemem << endl;
																																			cout << ",fp32_param_buf = "<< fp32_param_buf << endl;
																																			cout << ",k2kbuf = " 		<< k2kbuf << endl;
																																			cout << ",uint_param_buf = "<< uint_param_buf << endl;
																																			cout << "\n" << flush;
																																		}

	
	status = clEnqueueWriteBuffer(uload_queue, fp32_param_buf, 	CL_FALSE, 0, 16 * sizeof(float), fp32_params, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * sizeof(float), fp32_k2k, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, uint_param_buf,	CL_FALSE, 0,  8 * sizeof(uint),	 uint_params, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, mipmap_buf,		CL_FALSE, 0,  8*8* sizeof(uint), MipMap, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	
	status = clEnqueueWriteBuffer(uload_queue, basemem, 		CL_FALSE, 0, image_size_bytes, 	baseImage.data, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
																																		if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::allocatemem_chk4.2\n\n" << flush;
	float depth = 1/( obj["max_depth"].asFloat() - obj["min_depth"].asFloat() );
	float zero  = 0;
	float one   = 1;
																																		if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::allocatemem_chk4 \t Initial inverse depth = "<< depth <<"\n\n" << flush;
	for (int i=0; i<2; i++){
		status = clEnqueueFillBuffer(uload_queue, gxmem[i], &zero, sizeof(float), 0, mm_size_bytes_C4, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
		status = clEnqueueFillBuffer(uload_queue, gymem[i], &zero, sizeof(float), 0, mm_size_bytes_C4, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	}																																	if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::allocatemem_chk4.1\n\n" << flush;
	
	status = clEnqueueFillBuffer(uload_queue, cdatabuf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, hdatabuf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.9\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, img_sum_buf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.10\n"<< endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, se3_sum_mem, 	&zero, sizeof(float),   0, se3_sum_size_bytes,0, NULL, &writeEvt);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, depth_mem, 	&depth, sizeof(float),  0, mm_size_bytes_C1,  0, NULL, &writeEvt);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	clFlush(uload_queue); status = clFinish(uload_queue); 																				if (status != CL_SUCCESS)	{ cout << "\nclFinish(uload_queue)=" << status << checkerror(status) <<"\n"  << flush; exit_(status);}
																																		
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nRunCL::allocatemem_chk5\n\n" << flush;
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
																																		if(verbosity>local_verbosity_threshold) {
																																			DownloadAndSave_3Channel( 	basemem,	ss.str(), paths.at("basemem"),		image_size_bytes, 	baseImage_size, 	baseImage_type, false ); 	cout << "\nbasemem,"<< flush;
																																			DownloadAndSave( 			gxmem[0],	ss.str(), paths.at("gxmem[0]"), 	mm_size_bytes_C1, 	mm_Image_size, 		CV_32FC1, 		false , 1);	cout << "\ngxmem[0],"<< flush;
																																			DownloadAndSave( 			gxmem[1],	ss.str(), paths.at("gxmem[1]"), 	mm_size_bytes_C1, 	mm_Image_size, 		CV_32FC1, 		false , 1);	cout << "\ngxmem[1],"<< flush;
																																			DownloadAndSaveVolume(		cdatabuf, 	ss.str(), paths.at("cdatabuf"), 	mm_size_bytes_C1,	mm_Image_size, 		CV_32FC1,  		false , 1);	cout << "\ncdatabuf,"<< flush;
																																		}
	/*	// TODO  update and reactivate the old kernels
	// set kernelArg. NB "0 &k2kbuf" & "2 &imgmem" set in calcCostVol(..)
	res = clSetKernelArg(cost_kernel, 1, sizeof(cl_mem),  &basemem);		if(res!=CL_SUCCESS){cout<<"\nbasemem res= "   		<<checkerror(res)<<"\n"<<flush;exit_(res);} // base				
	res = clSetKernelArg(cost_kernel, 3, sizeof(cl_mem),  &cdatabuf);		if(res!=CL_SUCCESS){cout<<"\ncdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);} // cdata
	res = clSetKernelArg(cost_kernel, 4, sizeof(cl_mem),  &hdatabuf);		if(res!=CL_SUCCESS){cout<<"\nhdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);} // hdata
	res = clSetKernelArg(cost_kernel, 5, sizeof(cl_mem),  &lomem);			if(res!=CL_SUCCESS){cout<<"\nlomem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);} // lo
	res = clSetKernelArg(cost_kernel, 6, sizeof(cl_mem),  &himem);			if(res!=CL_SUCCESS){cout<<"\nhimem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);} // hi
	res = clSetKernelArg(cost_kernel, 7, sizeof(cl_mem),  &amem);			if(res!=CL_SUCCESS){cout<<"\namem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);} // a
	res = clSetKernelArg(cost_kernel, 8, sizeof(cl_mem),  &dmem);			if(res!=CL_SUCCESS){cout<<"\ndmem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);} // d
	res = clSetKernelArg(cost_kernel, 9, sizeof(cl_mem),  &fp16_param_buf);	if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "		<<checkerror(res)<<"\n"<<flush;exit_(res);} // param_buf
	res = clSetKernelArg(cost_kernel,10, sizeof(cl_mem),  &img_sum_buf);	if(res!=CL_SUCCESS){cout<<"\nimg_sum_buf res = " 	<<checkerror(res)<<"\n"<<flush;exit_(res);} // cdata
	*/
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nRunCL::allocatemem_chk6\n\n" << flush;
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
																																		if(verbosity>local_verbosity_threshold) cout << "RunCL::allocatemem_finished\n\n" << flush;
}


RunCL::~RunCL(){
																																		cout<<"\nRunCL::~RunCL_chk0_called"<<flush;
	/*
	cl_int status;
	status = clReleaseKernel(cost_kernel);      	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(cache3_kernel);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(updateQD_kernel);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseKernel(updateA_kernel);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }

	status = clReleaseProgram(m_program);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseCommandQueue(m_queue);		if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseCommandQueue(uload_queue);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseCommandQueue(dload_queue);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseCommandQueue(track_queue);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseContext(m_context);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; }
	*/
	cl_int status;
	// release memory
	for (int i=0; i<2; i++){
		status = clReleaseMemObject(imgmem[i]);				if (status != CL_SUCCESS)	{ cout << "\nimgmem["<<i<<"]             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(gxmem[i]);				if (status != CL_SUCCESS)	{ cout << "\ngxmem["<<i<<"]              status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(gymem[i]);				if (status != CL_SUCCESS)	{ cout << "\ngymem["<<i<<"]              status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(g1mem[i]);				if (status != CL_SUCCESS)	{ cout << "\ng1mem["<<i<<"]              status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(k_map_mem[i]);			if (status != CL_SUCCESS)	{ cout << "\nk_map_mem["<<i<<"]          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(dist_map_mem[i]);		if (status != CL_SUCCESS)	{ cout << "\ndist_map_mem["<<i<<"]       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(SE3_grad_map_mem[i]);	if (status != CL_SUCCESS)	{ cout << "\nSE3_grad_map_mem["<<i<<"]   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
	}
	status = clReleaseMemObject(keyframe_SE3_grad_map_mem);	if (status != CL_SUCCESS)	{ cout << "\nSE3_incr_map_mem status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
	
	status = clReleaseMemObject(SE3_incr_map_mem);	if (status != CL_SUCCESS)	{ cout << "\nSE3_incr_map_mem status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
	status = clReleaseMemObject(SE3_map_mem);		if (status != CL_SUCCESS)	{ cout << "\nSE3_map_mem      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
	status = clReleaseMemObject(basemem);			if (status != CL_SUCCESS)	{ cout << "\nbasemem          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.2"<<flush;
	status = clReleaseMemObject(depth_mem);			if (status != CL_SUCCESS)	{ cout << "\ndepth_mem        status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.2"<<flush;
	
	status = clReleaseMemObject(keyframe_basemem);	if (status != CL_SUCCESS)	{ cout << "\ndepth_mem        status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.2"<<flush;
	status = clReleaseMemObject(keyframe_g1mem);	if (status != CL_SUCCESS)	{ cout << "\ndepth_mem        status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.2"<<flush;
	
	status = clReleaseMemObject(dmem);				if (status != CL_SUCCESS)	{ cout << "\ndmem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.3"<<flush;
	status = clReleaseMemObject(amem);				if (status != CL_SUCCESS)	{ cout << "\namem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.4"<<flush;
	status = clReleaseMemObject(lomem);				if (status != CL_SUCCESS)	{ cout << "\nlomem            status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.5"<<flush;
	status = clReleaseMemObject(himem);				if (status != CL_SUCCESS)	{ cout << "\nhimem            status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.6"<<flush;
	status = clReleaseMemObject(qmem);				if (status != CL_SUCCESS)	{ cout << "\ndmem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.7"<<flush;
	status = clReleaseMemObject(cdatabuf);			if (status != CL_SUCCESS)	{ cout << "\ncdatabuf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.8"<<flush;
	status = clReleaseMemObject(hdatabuf);			if (status != CL_SUCCESS)	{ cout << "\nhdatabuf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.9"<<flush;
	
	status = clReleaseMemObject(img_sum_buf);		if (status != CL_SUCCESS)	{ cout << "\nhdatabuf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.10"<<flush;
	status = clReleaseMemObject(fp32_param_buf);	if (status != CL_SUCCESS)	{ cout << "\nfp32_param_buf   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.11"<<flush;
	status = clReleaseMemObject(k2kbuf);			if (status != CL_SUCCESS)	{ cout << "\nk2kbuf           status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.12"<<flush;
	
	status = clReleaseMemObject(SO3_k2kbuf);		if (status != CL_SUCCESS)	{ cout << "\nSE3_k2kbuf       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.13"<<flush;
	status = clReleaseMemObject(SE3_k2kbuf);		if (status != CL_SUCCESS)	{ cout << "\nSE3_k2kbuf       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.13"<<flush;
	status = clReleaseMemObject(uint_param_buf);	if (status != CL_SUCCESS)	{ cout << "\nuint_param_buf   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.14"<<flush;
	status = clReleaseMemObject(mipmap_buf);		if (status != CL_SUCCESS)	{ cout << "\nmipmap_buf       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.15"<<flush;
	status = clReleaseMemObject(gaussian_buf);		if (status != CL_SUCCESS)	{ cout << "\ngaussian_buf     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.16"<<flush;
	
	status = clReleaseMemObject(se3_sum_mem);		if (status != CL_SUCCESS)	{ cout << "\nse3_sum_mem      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.17"<<flush;
	status = clReleaseMemObject(se3_sum2_mem);		if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.18"<<flush;
	
	status = clReleaseMemObject(SE3_rho_map_mem	);	if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.19"<<flush;
	status = clReleaseMemObject(se3_sum_rho_sq_mem);if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.19.5"<<flush;
	
	status = clReleaseMemObject(img_stats_buf);		if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.20"<<flush;
	status = clReleaseMemObject(pix_sum_mem);		if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.21"<<flush;
	status = clReleaseMemObject(var_sum_mem);		if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.22"<<flush;
	
	// release kernels
	status = clReleaseKernel(cvt_color_space_linear_kernel);	if (status != CL_SUCCESS)	{ cout << "\ncvt_color_space_linear_kernel 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.23"<<flush;
	status = clReleaseKernel(img_variance_kernel);				if (status != CL_SUCCESS)	{ cout << "\nimg_variance_kernel 			status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.24"<<flush;
	status = clReleaseKernel(reduce_kernel);					if (status != CL_SUCCESS)	{ cout << "\nreduce_kernel 					status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.25"<<flush;
	status = clReleaseKernel(mipmap_linear_kernel);				if (status != CL_SUCCESS)	{ cout << "\nmipmap_linear_kernel 			status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.26"<<flush;
	status = clReleaseKernel(img_grad_kernel);					if (status != CL_SUCCESS)	{ cout << "\nimg_grad_kernel 				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.27"<<flush;
	status = clReleaseKernel(comp_param_maps_kernel);			if (status != CL_SUCCESS)	{ cout << "\ncomp_param_maps_kernel 		status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.28"<<flush;
	status = clReleaseKernel(se3_grad_kernel);					if (status != CL_SUCCESS)	{ cout << "\nse3_grad_kernel 				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.29"<<flush;
	
	status = clReleaseKernel(invert_depth_kernel);				if (status != CL_SUCCESS)	{ cout << "\nse3_grad_kernel 				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.29"<<flush;
	status = clReleaseKernel(depth_cost_vol_kernel);			if (status != CL_SUCCESS)	{ cout << "\nse3_grad_kernel 				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.29"<<flush;
	status = clReleaseKernel(updateQD_kernel);					if (status != CL_SUCCESS)	{ cout << "\nse3_grad_kernel 				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.29"<<flush;
	status = clReleaseKernel(updateG_kernel);					if (status != CL_SUCCESS)	{ cout << "\nse3_grad_kernel 				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.29"<<flush;
	status = clReleaseKernel(updateA_kernel);					if (status != CL_SUCCESS)	{ cout << "\nse3_grad_kernel 				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.29"<<flush;
	
	// release command queues
	status = clReleaseCommandQueue(m_queue);		if (status != CL_SUCCESS)	{ cout << "\nm_queue 		status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.30"<<flush;
	status = clReleaseCommandQueue(uload_queue);	if (status != CL_SUCCESS)	{ cout << "\nuload_queue 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.31"<<flush;
	status = clReleaseCommandQueue(dload_queue);	if (status != CL_SUCCESS)	{ cout << "\ndload_queue 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.32"<<flush;
	status = clReleaseCommandQueue(track_queue);	if (status != CL_SUCCESS)	{ cout << "\ntrack_queue 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.33"<<flush;
	
	// release Program
	clReleaseProgram(m_program);	if (status != CL_SUCCESS)	{ cout << "\nm_program 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.34"<<flush;
	
	// release context
	clReleaseContext(m_context);	if (status != CL_SUCCESS)	{ cout << "\nm_context 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.35"<<flush;
																																			cout<<"\nRunCL::~RunCL_chk1_finished"<<flush;
}

void RunCL::CleanUp(){// TODO do this with loops and #define names
																																			cout<<"\nRunCL::CleanUp_chk0"<<flush;
	/*
																																			cl_int status;
	// release memory
	for (int i=0; i<2; i++){
		status = clReleaseMemObject(imgmem[i]);				if (status != CL_SUCCESS)	{ cout << "\nimgmem["<<i<<"]             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(gxmem[i]);				if (status != CL_SUCCESS)	{ cout << "\ngxmem["<<i<<"]              status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(gymem[i]);				if (status != CL_SUCCESS)	{ cout << "\ngymem["<<i<<"]              status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(g1mem[i]);				if (status != CL_SUCCESS)	{ cout << "\ng1mem["<<i<<"]              status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(k_map_mem[i]);			if (status != CL_SUCCESS)	{ cout << "\nk_map_mem["<<i<<"]          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(dist_map_mem[i]);		if (status != CL_SUCCESS)	{ cout << "\ndist_map_mem["<<i<<"]       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
		status = clReleaseMemObject(SE3_grad_map_mem[i]);	if (status != CL_SUCCESS)	{ cout << "\nSE3_grad_map_mem["<<i<<"]   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
	}
	status = clReleaseMemObject(SE3_incr_map_mem);	if (status != CL_SUCCESS)	{ cout << "\nSE3_incr_map_mem status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
	status = clReleaseMemObject(SE3_map_mem);		if (status != CL_SUCCESS)	{ cout << "\nSE3_map_mem      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
	status = clReleaseMemObject(basemem);			if (status != CL_SUCCESS)	{ cout << "\nbasemem          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.2"<<flush;
	status = clReleaseMemObject(depth_mem);			if (status != CL_SUCCESS)	{ cout << "\ndepth_mem        status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.2"<<flush;
	status = clReleaseMemObject(dmem);				if (status != CL_SUCCESS)	{ cout << "\ndmem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.3"<<flush;
	status = clReleaseMemObject(amem);				if (status != CL_SUCCESS)	{ cout << "\namem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.4"<<flush;
	status = clReleaseMemObject(lomem);				if (status != CL_SUCCESS)	{ cout << "\nlomem            status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.5"<<flush;
	status = clReleaseMemObject(himem);				if (status != CL_SUCCESS)	{ cout << "\nhimem            status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.6"<<flush;
	status = clReleaseMemObject(qmem);				if (status != CL_SUCCESS)	{ cout << "\ndmem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.7"<<flush;
	
	status = clReleaseMemObject(cdatabuf);			if (status != CL_SUCCESS)	{ cout << "\ncdatabuf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.8"<<flush;
	status = clReleaseMemObject(hdatabuf);			if (status != CL_SUCCESS)	{ cout << "\nhdatabuf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.9"<<flush;
	status = clReleaseMemObject(img_sum_buf);		if (status != CL_SUCCESS)	{ cout << "\nhdatabuf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.10"<<flush;
	status = clReleaseMemObject(fp32_param_buf);	if (status != CL_SUCCESS)	{ cout << "\nfp32_param_buf   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.11"<<flush;
	status = clReleaseMemObject(k2kbuf);			if (status != CL_SUCCESS)	{ cout << "\nk2kbuf           status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.12"<<flush;
	status = clReleaseMemObject(SE3_k2kbuf);		if (status != CL_SUCCESS)	{ cout << "\nSE3_k2kbuf       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.13"<<flush;
	status = clReleaseMemObject(uint_param_buf);	if (status != CL_SUCCESS)	{ cout << "\nuint_param_buf   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.14"<<flush;
	status = clReleaseMemObject(mipmap_buf);		if (status != CL_SUCCESS)	{ cout << "\nmipmap_buf       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.15"<<flush;
	
	status = clReleaseMemObject(gaussian_buf);		if (status != CL_SUCCESS)	{ cout << "\ngaussian_buf     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.16"<<flush;
	status = clReleaseMemObject(se3_sum_mem);		if (status != CL_SUCCESS)	{ cout << "\nse3_sum_mem      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.17"<<flush;
	status = clReleaseMemObject(se3_sum2_mem);		if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.18"<<flush;
	status = clReleaseMemObject(SE3_rho_map_mem	);	if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.19"<<flush;
	status = clReleaseMemObject(se3_sum_rho_sq_mem);if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.19.5"<<flush;
	status = clReleaseMemObject(img_stats_buf);		if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.20"<<flush;
	status = clReleaseMemObject(pix_sum_mem);		if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.21"<<flush;
	status = clReleaseMemObject(var_sum_mem);		if (status != CL_SUCCESS)	{ cout << "\nreduce_param_buf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.22"<<flush;
	

	// release kernels
	status = clReleaseKernel(cvt_color_space_linear_kernel);	if (status != CL_SUCCESS)	{ cout << "\ncvt_color_space_linear_kernel 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.23"<<flush;
	status = clReleaseKernel(img_variance_kernel);				if (status != CL_SUCCESS)	{ cout << "\nimg_variance_kernel 			status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.24"<<flush;
	status = clReleaseKernel(reduce_kernel);					if (status != CL_SUCCESS)	{ cout << "\nreduce_kernel 					status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.25"<<flush;
	status = clReleaseKernel(mipmap_linear_kernel);				if (status != CL_SUCCESS)	{ cout << "\nmipmap_linear_kernel 			status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.26"<<flush;
	status = clReleaseKernel(img_grad_kernel);					if (status != CL_SUCCESS)	{ cout << "\nimg_grad_kernel 				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.27"<<flush;
	status = clReleaseKernel(comp_param_maps_kernel);			if (status != CL_SUCCESS)	{ cout << "\ncomp_param_maps_kernel 		status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.28"<<flush;
	status = clReleaseKernel(se3_grad_kernel);					if (status != CL_SUCCESS)	{ cout << "\nse3_grad_kernel 				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.29"<<flush;
	
	// release command queues
	status = clReleaseCommandQueue(m_queue);		if (status != CL_SUCCESS)	{ cout << "\nm_queue 		status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.30"<<flush;
	status = clReleaseCommandQueue(uload_queue);	if (status != CL_SUCCESS)	{ cout << "\nuload_queue 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.31"<<flush;
	status = clReleaseCommandQueue(dload_queue);	if (status != CL_SUCCESS)	{ cout << "\ndload_queue 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.32"<<flush;
	status = clReleaseCommandQueue(track_queue);	if (status != CL_SUCCESS)	{ cout << "\ntrack_queue 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.33"<<flush;
	
	// release Program
	clReleaseProgram(m_program);	if (status != CL_SUCCESS)	{ cout << "\nm_program 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.34"<<flush;
	
	// release context
	clReleaseContext(m_context);	if (status != CL_SUCCESS)	{ cout << "\nm_context 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.35"<<flush;
																																			cout<<"\nRunCL::CleanUp_chk1_finished"<<flush;
																																			
	*/
}

void RunCL::exit_(cl_int res)
{
	//CleanUp();
	//~RunCL(); Never need to call a destructor manually.
	exit(res);
}

