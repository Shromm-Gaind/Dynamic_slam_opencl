#include "RunCL.h"

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

using namespace std;

RunCL::RunCL(Json::Value obj_)
{
	obj = obj_;
	verbosity = obj["verbosity"].asInt();
	std::cout << "RunCL::RunCL verbosity = " << verbosity << std::flush;
																							if(verbosity>0) cout << "\nRunCL_chk 0\n" << flush;
	createFolders( );																		/*Step1: Getting platforms and choose an available one.*/////////
	cl_uint 		numPlatforms;															//the NO. of platforms
	cl_platform_id 	platform 		= NULL;													//the chosen platform
	cl_int			status 			= clGetPlatformIDs(0, NULL, &numPlatforms);				if (status != CL_SUCCESS){ cout << "Error: Getting platforms!" << endl; exit_(status); }
	uint			conf_platform	= obj["opencl_platform"].asUInt();						if(verbosity>0) cout << "numPlatforms = " << numPlatforms << "\n" << flush;
	
	if (numPlatforms > conf_platform){														/*Choose the platform.*/
		cl_platform_id* platforms 	= (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
		status 	 					= clGetPlatformIDs(numPlatforms, platforms, NULL);		if (status != CL_SUCCESS){ cout << "Error: Getting platformsIDs" << endl; exit_(status); }
		platform 					= platforms[ conf_platform ];
		free(platforms);																	if(verbosity>0) cout << "\nplatforms[0] = "<<platforms[0]<<", \nplatforms[1] = "<<platforms[1]\
																							<<"\nSelected platform number :"<<conf_platform<<", cl_platform_id platform = " << platform<<"\n"<<flush;
	} else {cout<<"Platform num "<<conf_platform<<" not available."<<flush; exit(0);}
	
	cl_uint			numDevices		= 0;													/*Step 2:Query the platform.*//////////////////////////////////
	cl_device_id    *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);			if (status != CL_SUCCESS) {cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	uint conf_device = obj["opencl_device"].asUInt();
	
	if (numDevices > conf_device){															/*Choose the device*/
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status  = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}																						if (status != CL_SUCCESS) {cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);} 

																							if(verbosity>0) cout << "\nRunCL_chk 3\n" << flush; cout << "cl_device_id  devices = " << devices << "\n" << flush;
	
	cl_context_properties cps[3]={CL_CONTEXT_PLATFORM,(cl_context_properties)platform,0};	/*Step 3: Create context.*////////////////////////////////////
	m_context 	= clCreateContextFromType( cps, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);	if(status!=0) 			{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	
	deviceId  	= devices[conf_device];														/*Step 4: Create command queue & associate context.*///////////
	cl_command_queue_properties prop[] = { 0 };												//  NB Device (GPU) queues are out-of-order execution -> need synchronization.
	m_queue 	= clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	uload_queue = clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	dload_queue = clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	track_queue = clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);}
																							// Multiple queues for latency hiding: Upload, Download, Mapping, Tracking,... autocalibration, SIRFS, SPMP
																							// NB Might want to create command queues on multiple platforms & devices.
																							// NB might want to divde a task across multiple MPI Ranks on a multi-GPU WS or cluster.
	
	const char *filename = obj["kernel_filepath"].asCString();								/*Step 5: Create program object*///////////////////////////////
	string sourceStr;
	status 						= convertToString(filename, sourceStr);						if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	const char 	*source 		= sourceStr.c_str();
	size_t 		sourceSize[] 	= { strlen(source) };
	m_program 	= clCreateProgramWithSource(m_context, 1, &source, sourceSize, NULL);
	
	status = clBuildProgram(m_program, 1, devices, NULL, NULL, NULL);						/*Step 6: Build program.*/////////////////////
	if (status != CL_SUCCESS){
		printf("\nclBuildProgram failed: %d\n", status);
		char buf[0x10000];
		clGetProgramBuildInfo(m_program, deviceId, CL_PROGRAM_BUILD_LOG, 0x10000, buf, NULL);
		printf("\n%s\n", buf);
		exit_(status);
	}
	
	//cost_kernel     = clCreateKernel(m_program, "BuildCostVolume2", 	NULL);					/*Step 7: Create kernel objects.*//////////// TODO update and reactivate these kernels
	//cache3_kernel   = clCreateKernel(m_program, "CacheG3", 			NULL);
	//updateQD_kernel = clCreateKernel(m_program, "UpdateQD", 			NULL);
	//updateA_kernel  = clCreateKernel(m_program, "UpdateA2", 			NULL);
	
	cvt_color_space_kernel 	= clCreateKernel(m_program, "cvt_color_space", 		NULL);		// New kernels
	mipmap_kernel			= clCreateKernel(m_program, "mipmap", 				NULL);
	img_grad_kernel			= clCreateKernel(m_program, "img_grad", 			NULL);
	
	basemem=imgmem=cdatabuf=hdatabuf=k2kbuf=dmem=amem=basegraymem=gxmem=gymem=g1mem=lomem=himem=0;		// set device pointers to zero
																							if(verbosity>0) cout << "RunCL_constructor finished\n" << flush;
}

void RunCL::createFolders(){
																						if(verbosity>0) cout << "\n createFolders_chk 0\n" << flush;
	std::time_t   result  = std::time(nullptr);
	std::string   out_dir = std::asctime(std::localtime(&result));
	out_dir.pop_back(); 																// req to remove new_line from end of string.
	
	boost::filesystem::path 	out_path(boost::filesystem::current_path());
	boost::filesystem::path 	conf_outpath( obj["outpath"].asString() );
	if (conf_outpath.empty() ||  conf_outpath.is_absolute() ) {
		out_path = out_path.parent_path().parent_path();								// move "out_path" up two levels in the directory tree.
		out_path += conf_outpath;
	}else {out_path = conf_outpath;}
	out_path += "/output/";
	if(boost::filesystem::create_directory(out_path)) { std::cerr<< "Directory Created: "<<out_path<<std::endl;}else{ std::cerr<< "Output directory previously created: "<<out_path<<std::endl;}
	out_path +=  out_dir;																if(verbosity>0) cout <<"Creating output sub-directories: "<< out_path <<std::endl;
	boost::filesystem::create_directory(out_path);
	out_path += "/";																	if(verbosity>0) cout << "\n createFolders_chk 1\n" << flush;
	
	boost::filesystem::path temp_path = out_path;								// Vector of device buffer names
	std::vector<std::string> names = {"basemem","imgmem","cdatabuf","hdatabuf","pbuf","dmem", "amem","basegraymem","gxmem","gymem","g1mem","qmem","lomem","himem", "img_sum_buf"};
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
		cout << "\nRunCL::createFolders() chk1\n";
		cout << "KEY\tPATH\n";														// print the folder paths
		for (auto itr=paths.begin(); itr!=paths.end(); ++itr) { cout<<"First:["<< itr->first << "]\t:\t Second:"<<itr->second<<"\n"; }
		cout<<"\npaths.at(\"basemem\")="<<paths.at("basemem")<<"\n"<<flush;
	}
}

void RunCL::DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range ){
																				//if(verbosity>0) cout<<"\n\nDownloadAndSave chk0"<<flush;
																				if(verbosity>0) cout<<"\n\nDownloadAndSave filename = ["<<folder_tiff.filename().string()<<"] "<<flush;
																				/*
																				cout <<", folder="<<folder_tiff<<flush;
																				cout <<", image_size_bytes="<<image_size_bytes<<flush;
																				cout <<", size_mat="<<size_mat<<flush;
																				cout <<", type_mat="<<size_mat<<"\t"<<flush;
																				*/
		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);					// (int rows, int cols, int type)
		ReadOutput(temp_mat.data, buffer,  image_size_bytes); 					// NB contains elements of type_mat, (CV_32FC1 for most buffers)
																				if(verbosity>1) cout<<"\n\nDownloadAndSave finished ReadOutput\n\n"<<flush;
		if (temp_mat.type() == CV_16FC1)	temp_mat.convertTo(temp_mat, CV_32FC1);	// NB conversion to FP32 req for cv::sum(..).	
		cv::Scalar sum = cv::sum(temp_mat);										// NB always returns a 4 element vector.

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
																				if(verbosity>1) cout<<"\n\nDownloadAndSave filename = ["<<ss.str()<<"]";
		cv::Mat outMat;
		if (type_mat != CV_32FC1 && type_mat != CV_16FC1 ) {
			cout << "\n\n## Error  (type_mat != CV_32FC1 or CV_16FC1) ##\n\n" << flush;
			return;
		}
		if (max_range == 0){ temp_mat /= maxVal;}								// Squash/stretch & shift to 0.0-1.0 range
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

void RunCL::DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show ){
																				if(verbosity>0) cout<<"\n\nDownloadAndSave_3Channel filename = ["<<folder_tiff.filename()<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
		cv::Mat temp_mat, temp_mat2;
		
		if (type_mat == CV_16FC3)	{
			temp_mat2 = cv::Mat::zeros (size_mat, CV_16FC3);
			//cout << "\nReading CV_16FC3. size_mat="<< size_mat<<",   temp_mat2.total()*temp_mat2.elemSize()="<< temp_mat2.total()*temp_mat2.elemSize() << flush;
			
			ReadOutput(temp_mat2.data, buffer,  temp_mat2.total()*temp_mat2.elemSize() );  // baseImage.total() * baseImage.elemSize()
			
			temp_mat = cv::Mat::zeros (size_mat, CV_32FC3);
			temp_mat2.convertTo(temp_mat, CV_32FC3);	// NB conversion to FP32 req for cv::sum(..).
			
		} else {
			temp_mat = cv::Mat::zeros (size_mat, type_mat);
			ReadOutput(temp_mat.data, buffer,  image_size_bytes);
		}
		
		cv::Scalar 	sum = cv::sum(temp_mat);									// NB always returns a 4 element vector.
		string 		type_string=checkCVtype(type_mat);
		double 		minVal[3]={1,1,1}, 					maxVal[3]={0,0,0};
		cv::Point 	minLoc[3]={{0,0},{0,0},{0,0}}, 		maxLoc[3]={{0,0},{0,0},{0,0}};
		vector<cv::Mat> spl;
		split(temp_mat, spl);													// process - extract only the correct channel
		double max = 0;
		for (int i =0; i < 3; ++i){
			cv::minMaxLoc(spl[i], &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i]);
			if (maxVal[i] > max) max = maxVal[i];
		}
		stringstream ss;
		stringstream png_ss;
		ss<<"/"<<folder_tiff.filename().string()<<"_"<<count<<"_sum"<<sum<<"type_"<<type_string<<"min("<<minVal[0]<<","<<minVal[1]<<","<<minVal[2]<<")_max("<<maxVal[0]<<","<<maxVal[1]<<","<<maxVal[2]<<")";
		png_ss<< "/" << folder_tiff.filename().string() << "_" << count;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);									// NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
		}
		boost::filesystem::path folder_png = folder_tiff;
		folder_png  += "/png/";
		folder_png  += png_ss.str();
		folder_png  += ".png";

		folder_tiff += ss.str();
		folder_tiff += ".tiff";

		cv::Mat outMat;
		if ((type_mat == CV_32FC3) || (type_mat == CV_32FC4)){
			cv::imwrite(folder_tiff.string(), temp_mat );
			temp_mat *=256;
			temp_mat.convertTo(outMat, CV_8U);
			if (type_mat == CV_32FC4){
				std::vector<cv::Mat> matChannels;
				cv::split(outMat, matChannels);
				matChannels.at(3)=255;
				cv::merge(matChannels, outMat);
			}
			cv::imwrite(folder_png.string(), (outMat) );					// Has "Grayscale 16-bit gamma integer"
		}else if (type_mat == CV_8UC3){
			cv::imwrite(folder_tiff.string(), temp_mat );
			cv::imwrite(folder_png.string(),  temp_mat );
		}else if (type_mat == CV_16FC3) {									// This demonstrates that <cv::float16_t> != <cl_half> and the read/write up/download of these types needs more debugging. NB Cannot use <cv::float16_t>  to prepare  <cl_half> data to the GPU.
			//cout << "\n Writing CV_16FC3 to .tiff & .png .\n"<< flush;
			//cout << "\n temp_mat2.at<cv::float16_t>(101,100-105) = " << temp_mat2.at<cv::float16_t>(101,100) << "," << temp_mat2.at<cv::float16_t>(101,101) << ","<< temp_mat2.at<cv::float16_t>(101,102) << ","<< temp_mat2.at<cv::float16_t>(101,103) << ","<< temp_mat2.at<cv::float16_t>(101,104) << ","<< temp_mat2.at<cv::float16_t>(101,105) << ","<< flush; 
			//cout << "\n temp_mat2.at<cl_half>(101,100-105) = " << temp_mat2.at<cl_half>(101,100) << "," << temp_mat2.at<cl_half>(101,101) << ","<< temp_mat2.at<cl_half>(101,102) << ","<< temp_mat2.at<cl_half>(101,103) << ","<< temp_mat2.at<cl_half>(101,104) << ","<< temp_mat2.at<cl_half>(101,105) << ","<< flush; 
			//cout << "\n temp_mat2.at<cl_half>(101,100) x,y,z,w,s0,s3 = " << temp_mat2.at<cl_half3>(101,100).x << "," << temp_mat2.at<cl_half3>(101,100).y << ","<< temp_mat2.at<cl_half3>(101,100).z << ","<< temp_mat2.at<cl_half3>(101,100).w << ","<< temp_mat2.at<cl_half3>(101,100).s0 << ","<< temp_mat2.at<cl_half3>(101,100).s3 << ","<< flush; 
			temp_mat2 *=256;
			cv::imwrite(folder_tiff.string(), temp_mat2 );
			
			temp_mat2.convertTo(outMat, CV_8UC3);
			cv::imwrite(folder_png.string(), (outMat) );
		}else {cout << "\n\nError RunCL::DownloadAndSave_3Channel(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}
}

void RunCL::DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range ){
																				if(verbosity>0) {
																					cout<<"\n\nDownloadAndSaveVolume, costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]";
																					cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<size_mat<<"\t"<<flush;
																				}
	

	for(int i=0; i<costVolLayers; i++){
																				if(verbosity>1) cout << "\ncostVolLayers="<<costVolLayers<<", i="<<i<<"\t";
		size_t offset = i * image_size_bytes;
		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);					//(int rows, int cols, int type)
		
		ReadOutput(temp_mat.data, buffer,  image_size_bytes, offset);
																				if(verbosity>1) cout << "\nRunCL::DownloadAndSaveVolume, ReadOutput completed\n"<<flush;
		if (temp_mat.type() == CV_16FC1)	temp_mat.convertTo(temp_mat, CV_32FC1);	// NB conversion to FP32 req for cv::sum(..).
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
			temp_mat.convertTo(temp, CV_8U);									// NB need CV_U8 for imshow(..)
			cv::imshow(ss.str(), temp);
		}
		new_filepath += ss.str();
		new_filepath += ".tiff";
		folder_png += "/png/";
		folder_png += png_ss.str();
		folder_png += ".png";
																				if(verbosity>1) cout << "\nnew_filepath.string() = "<<new_filepath.string() <<"\n";
		cv::Mat outMat;

		if (type_mat != CV_32FC1 && type_mat != CV_16FC1 ) {
			cout << "\n\n## Error  (type_mat != CV_32FC1 or CV_16FC1) ##\n\n" << flush;
			return;
		}
		if (max_range == 0){ temp_mat /= maxVal;}								// Squash/stretch & shift to 0.0-1.0 range
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
/*
void RunCL::computeSigmas(float epsilon, float theta, float L, cv::float16_t &sigma_d, cv::float16_t &sigma_q ){
		float mu	= 2.0*std::sqrt((1.0/theta)*epsilon) /L;
		sigma_d		= cv::float16_t( mu / (2.0/ theta)  );
		sigma_q 	= cv::float16_t( mu / (2.0*epsilon) );
}
*/
void RunCL::computeSigmas(float epsilon, float theta, float L, float &sigma_d, float &sigma_q ){
		float mu	= 2.0*std::sqrt((1.0/theta)*epsilon) /L;
		sigma_d		=  mu / (2.0/ theta)  ;
		sigma_q 	=  mu / (2.0*epsilon) ;
}

void RunCL::computeSigmas(float epsilon, float theta, float L, cl_half &sigma_d, cl_half &sigma_q ){
		float mu	= 2.0*std::sqrt((1.0/theta)*epsilon) /L;
		sigma_d		= cl_half( mu / (2.0/ theta)  );
		sigma_q 	= cl_half( mu / (2.0*epsilon) );
}

void RunCL::allocatemem()//float* gx, float* gy, float* params, int layers, cv::Mat &baseImage, float *cdata, float *hdata, float *img_sum_data)
{
																																		if(verbosity>0) cout << "\n\nRunCL::allocatemem_chk0\n\n" << flush;
																																		if(baseImage.empty()){cout <<"\nError RunCL::allocatemem() : runcl.baseImage.empty()"<<flush; exit(0); }
	stringstream 		ss;
	ss << "allocatemem";
	cl_int 				status;
	cl_event 			writeEvt;
	
	image_size_bytes	= baseImage.total() * baseImage.elemSize() ;
	costVolLayers 		= 2*( 1 + obj["layers"].asUInt() );
	baseImage_size 		= baseImage.size();
	baseImage_type 		= baseImage.type();
	width				= baseImage.cols;
	height				= baseImage.rows;
	layerstep 			= width * height;
	fp16_size			= sizeof(cl_half);
	
	mm_margin			= obj["MipMap_margin"].asUInt();																				// MipMap parameters
	mm_width 			= baseImage.cols * 1.5 + 4 * mm_margin;
	mm_height 			= baseImage.rows       + 2 * mm_margin;
	mm_layerstep		= mm_width * mm_height;
	
	cv::Mat temp(mm_height, mm_width, CV_32FC3);
	mm_Image_size		= temp.size();
	mm_Image_type		= temp.type();
	mm_size_bytes_C3	= temp.total() * temp.elemSize() ;																				// for mipmaps with CV_16FC3  mm_width * mm_height * fp16_size;	// for FP16 'half', or BF16 on Tensor cores
	mm_size_bytes_C4	= temp.total() * 4 * sizeof(float);
	mm_size_bytes_half4	= temp.total() * 4 * fp16_size;																					// for mipmaps with half4
	
	cv::Mat temp2(mm_height, mm_width, CV_32FC1);
	mm_size_bytes_C1	= temp2.total() * temp2.elemSize();
	mm_vol_size_bytes	= mm_size_bytes_C1 * costVolLayers;
																																		if(verbosity>1) cout << "\n\nRunCL::allocatemem_chk1\n\n" << flush;
																																		// Get the maximum work group size for executing the kernel on the device ///////// From https://github.com/rsnemmen/OpenCL-examples/blob/e2c34f1dfefbd265cfb607c2dd6c82c799eb322a/square_array/square.c
	status = clGetKernelWorkGroupInfo(cvt_color_space_kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size), &local_work_size, NULL); 	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
																																		if(verbosity>1) cout << "\n\nRunCL::allocatemem_chk1.1,\t local_work_size="<< local_work_size <<"\n\n" << flush;
																																		// Number of total work items, calculated here after 1st image is loaded &=> know the size.
																																		// NB localSize must be devisor
																																		// NB global_work_size must be a whole number of "Preferred work group size multiple" for Nvidia. i.e. global_work_size should be slightly more than the number of point or pixels to be processed.
	global_work_size 	= ceil( (float)layerstep/(float)local_work_size ) * local_work_size;
	mm_global_work_size = ceil( (float)mm_layerstep/(float)local_work_size ) * local_work_size;
																																		if(verbosity>1){ 
																																			cout << "\n\nRunCL::allocatemem_chk1.2,\t global_work_size="<<global_work_size<<",\t mm_global_work_size="<<mm_global_work_size <<"\n" << flush;
																																			cout << "layerstep="<<layerstep <<",\t mm_layerstep"<< mm_layerstep<<"\n\n" << flush;
																																		}
																																		//local_work_size=32; // trial for nvidia
																																		if(verbosity>1){
																																			cout << "\n\nRunCL::allocatemem_chk2\n\n" << flush;
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
	cl_int res;
	imgmem				= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, mm_size_bytes_C4,  0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // MipMap in 'half' FP16.
	basemem				= clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, image_size_bytes,  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);} // Original image CV_8UC3
	
	dmem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	amem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gxmem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gymem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	g1mem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	lomem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	himem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	qmem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE						, 2 * mm_size_bytes_C1, 0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	cdatabuf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_vol_size_bytes, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	hdatabuf 			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_vol_size_bytes, 	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	img_sum_buf 		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, 2 * mm_vol_size_bytes,0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	// float debug buffer.
	
	//half_param_buf	= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 16 * fp16_size,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	//fp16_param_buf	= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 16 * fp16_size,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	//k2kbuf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 16 * fp16_size,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	fp32_param_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 16 * sizeof(float),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	k2kbuf				= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 16 * sizeof(float),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	uint_param_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 8 * sizeof(uint),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	mipmap_buf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 8 * sizeof(uint),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	//gaussian_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 9 * fp16_size,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//  TODO load gaussian kernel & size from conf.json .
	gaussian_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, 9 * sizeof(float),  	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//  TODO load gaussian kernel & size from conf.json .
	
																																		if(verbosity>1) {
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
																																			//cout << ",fp16_param_buf = "<< fp16_param_buf << endl;
																																			cout << ",fp32_param_buf = "<< fp32_param_buf << endl;
																																			cout << ",k2kbuf = " 		<< k2kbuf << endl;
																																			cout << ",uint_param_buf = "<< uint_param_buf << endl;
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
	cv::Mat cost 		= cv::Mat::ones (costVolLayers, mm_height * mm_width, CV_32FC1); 	//cost	= obj["initialCost"].asFloat()   ;	cost.convertTo(		cost,		temp.type() );	// Initialization of buffers. ? Are OpenCL buffers initialiyzed to zero by default ? TODO fix initialization CV_16FC1, poss with a kernel.
	cv::Mat hit      	= cv::Mat::zeros(costVolLayers, mm_height * mm_width, CV_32FC1); 	//hit	= obj["initialWeight"].asFloat() ;	hit.convertTo(		hit,		temp.type() );
	cv::Mat img_sum		= cv::Mat::zeros(costVolLayers, mm_height * mm_width, CV_32FC1);												//img_sum.convertTo(	img_sum,	temp.type() );
	cv::Mat gxy			= cv::Mat::ones (mm_height, mm_width, CV_32FC1);
																																		cout << "\n\nmm_vol_size_bytes="<<mm_vol_size_bytes<< ",\t cost.total()*cost.elemSize()="<< cost.total()*cost.elemSize() <<" .\n\n"<<flush; 
																																		//cv::imshow("cost", cost); cv::waitKey(5000);
	/*
	fp16_params[MAX_INV_DEPTH]	= cv::float16_t( 1/obj["min_depth"].asFloat()		);													// This works: Initialize 'params[]' from conf.json . # TODO ? separate 16 bit uint params ? to avoid floor() conversions.
	fp16_params[MIN_INV_DEPTH]	= cv::float16_t( 1/obj["max_depth"].asFloat()		);
				//INV_DEPTH_STEP	;
	fp16_params[ALPHA_G]		= cv::float16_t(   obj["alpha_g"].asFloat()			);
	fp16_params[BETA_G]			= cv::float16_t(   obj["beta_g"].asFloat()			);
	fp16_params[EPSILON]		= cv::float16_t(   obj["epsilon"].asFloat()			);
				//SIGMA_Q ;
				//SIGMA_D ;
	fp16_params[THETA]			= cv::float16_t(   obj["thetaStart"].asFloat()		);
	fp16_params[LAMBDA]			= cv::float16_t(   obj["lambda"].asFloat()			);
	fp16_params[SCALE_EAUX]		= cv::float16_t(   obj["scale_E_aux"].asFloat()		);
	*/
	
	fp32_params[MAX_INV_DEPTH]	=  1/obj["min_depth"].asFloat()		;													// This works: Initialize 'params[]' from conf.json . 
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
	
	
	/* 
	cl_half_params[MAX_INV_DEPTH]	= cl_half( 1/obj["min_depth"].asFloat()		);														// Does not work : Initialize 'params[]' from conf.json . # TODO ? separate 16 bit uint params ? to avoid floor() conversions.
	cl_half_params[MIN_INV_DEPTH]	= cl_half( 1/obj["max_depth"].asFloat()		);
	//INV_DEPTH_STEP	;
	cl_half_params[ALPHA_G]			= cl_half(   obj["alpha_g"].asFloat()		);
	cl_half_params[BETA_G]			= cl_half(   obj["beta_g"].asFloat()		);
	cl_half_params[EPSILON]			= cl_half(   obj["epsilon"].asFloat()		);
	//SIGMA_Q ;
	//SIGMA_D ;
	cl_half_params[THETA]			= cl_half(   obj["thetaStart"].asFloat()	);
	cl_half_params[LAMBDA]			= cl_half(   obj["lambda"].asFloat()		);
	cl_half_params[SCALE_EAUX]		= cl_half(   obj["scale_E_aux"].asFloat()	);
	*/
/*
																																		if(verbosity>1){
																																			float test_float = 12.06;							// NB half has approximately 3 decimal significat figures, and +/-5 decimal orders of magnitude
																																			cl_half test_half = cl_half(test_float);
																																			cout << "\n\nTesting cl_half:"<<flush;
																																			cout << "\ntest_float="<<test_float<<flush;
																																			cout << "\ntest_half="<<test_half<<flush;
																																			cout << "\nfloat(test_half)="<<float(test_half)<<flush;
																																			test_half = test_float;
																																			cout << "\ntest_half="<<test_half<<flush;
																																			cout << "\n"<<flush;
																																			test_half=cv::float16_t(test_float);
																																			cout << "\ntest_half="<<test_half<<flush;
																																			cv::float16_t test_cvfp16 = cv::float16_t(test_float);
																																			cout << "\ntest_cvfp16="<<test_cvfp16<<flush;
																																			cout << "\ntest_cvfp16="<<float(test_cvfp16)<<flush;
																																			cout << "\n";
																																			cout << "Oct content, test_float, test_half, test_cvfp16\n";
																																			cout << std::oct << test_float << endl;
																																			cout << std::oct << test_half << endl;
																																			cout << std::oct << test_cvfp16 << endl;
																																			cout << "\n\n"<<flush;
																																		}
*/
	uint_params[PIXELS]			= 	baseImage.rows * baseImage.cols ;
	uint_params[ROWS]			= 	baseImage.rows ;
	uint_params[COLS]			= 	baseImage.cols ;
	uint_params[LAYERS]			= 	obj["layers"].asUInt() ;
	uint_params[MARGIN]			= 	mm_margin ;
	uint_params[MM_PIXELS]		= 	mm_height * mm_width ;
	uint_params[MM_ROWS]		= 	mm_height ;
	uint_params[MM_COLS]		= 	mm_width ;
	
	computeSigmas( obj["epsilon"].asFloat(), obj["thetaStart"].asFloat(), obj["L"].asFloat(), fp32_params[SIGMA_Q], fp32_params[SIGMA_D] );
																																		//computeSigmas( obj["epsilon"].asFloat(), obj["thetaStart"].asFloat(), obj["L"].asFloat(), cl_half_params[SIGMA_Q], cl_half_params[SIGMA_D] );
																																		if(verbosity>1){
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
																																			cout << "\n";
																																			/*
																																			cout << "\n\nChecking cl_half_params[]";
																																			cout << "\ncl_half_params[MAX_INV_DEPTH]="	<<cl_half_params[MAX_INV_DEPTH]		<<"\t\t1/obj[\"min_depth\"].asFloat()="	<<1/obj["min_depth"].asFloat();
																																			cout << "\ncl_half_params[MIN_INV_DEPTH]="	<<cl_half_params[MIN_INV_DEPTH]		<<"\t\t1/obj[\"max_depth\"].asFloat()="	<<1/obj["max_depth"].asFloat();
																																			cout << "\ncl_half_params[INV_DEPTH_STEP]="	<<cl_half_params[INV_DEPTH_STEP];
																																			cout << "\ncl_half_params[ALPHA_G]="		<<cl_half_params[ALPHA_G]			<<"\t\tobj[\"alpha_g\"].asFloat()="		<<obj["alpha_g"].asFloat();
																																			cout << "\ncl_half_params[BETA_G]="			<<cl_half_params[BETA_G]			<<"\t\tobj[\"beta_g\"].asFloat()="		<<obj["beta_g"].asFloat();
																																			cout << "\ncl_half_params[EPSILON]="		<<cl_half_params[EPSILON]			<<"\t\tobj[\"epsilon\"].asFloat()="		<<obj["epsilon"].asFloat();
																																			cout << "\ncl_half_params[SIGMA_Q]="		<<cl_half_params[SIGMA_Q];
																																			cout << "\ncl_half_params[SIGMA_D]="		<<cl_half_params[SIGMA_D];
																																			cout << "\ncl_half_params[THETA]="			<<cl_half_params[THETA]				<<"\t\tobj[\"thetaStart\"].asFloat()="	<<obj["thetaStart"].asFloat();
																																			cout << "\ncl_half_params[LAMBDA]="			<<cl_half_params[LAMBDA]			<<"\t\tobj[\"lambda\"].asFloat()="		<<obj["lambda"].asFloat();
																																			cout << "\ncl_half_params[SCALE_EAUX]="		<<cl_half_params[SCALE_EAUX]		<<"\t\tobj[\"scale_E_aux\"].asFloat()="	<<obj["scale_E_aux"].asFloat();
																																			cout << "\n";
																																			*/
																																		}
	/*
	fp16_k2k[0]  = cv::float16_t( 1.0 );																									// initialize fp16_k2k as 'unity' transform, i.e. zero rotation & zero translation.
	fp16_k2k[5]  = cv::float16_t( 1.0 );
	fp16_k2k[10] = cv::float16_t( 1.0 );
	*/
	
	fp32_k2k[0]  =  1.0 ;																									// initialize fp16_k2k as 'unity' transform, i.e. zero rotation & zero translation.
	fp32_k2k[5]  =  1.0 ;
	fp32_k2k[10] =  1.0 ;
	
	/*
	cl_half_k2k[0]  = cl_half( 1.0 );																									// initialize cl_half_k2k as 'unity' transform, i.e. zero rotation & zero translation.
	cl_half_k2k[5]  = cl_half( 1.0 );
	cl_half_k2k[10] = cl_half( 1.0 );
	*/
																																		if(verbosity>1) cout << "\n\nRunCL::allocatemem_chk4\n\n" << flush;
	status = clEnqueueWriteBuffer(uload_queue, gxmem, 			CL_FALSE, 0, mm_size_bytes_C1, 	gxy.data, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, gymem, 			CL_FALSE, 0, mm_size_bytes_C1, 	gxy.data, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	//status = clEnqueueWriteBuffer(uload_queue, fp16_param_buf, 	CL_FALSE, 0, 16 * fp16_size, 	fp16_params, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	//status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * fp16_size, 	fp16_k2k, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, fp32_param_buf, 	CL_FALSE, 0, 16 * sizeof(float), fp32_params, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * sizeof(float), fp32_k2k, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, uint_param_buf,	CL_FALSE, 0, 8 * sizeof(uint),	uint_params, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, cdatabuf, 		CL_FALSE, 0, mm_vol_size_bytes, cost.data, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, hdatabuf, 		CL_FALSE, 0, mm_vol_size_bytes, hit.data, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.9\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, img_sum_buf, 	CL_FALSE, 0, mm_vol_size_bytes, img_sum.data, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.10\n"<< endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, basemem, 		CL_FALSE, 0, image_size_bytes, 	baseImage.data, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	clFlush(uload_queue); status = clFinish(uload_queue); 																				if (status != CL_SUCCESS)	{ cout << "\nclFinish(uload_queue)=" << status << checkerror(status) <<"\n"  << flush; exit_(status);}
																																		
																																		if(verbosity>1) {
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
																																		if(verbosity>1) {
																																			DownloadAndSave_3Channel( 	basemem,	ss.str(), paths.at("basemem"),		image_size_bytes, 	baseImage_size, 	baseImage_type, false );		// DownloadAndSave_3Channel(basemem,..) verify uploads.
																																			DownloadAndSave( 			gxmem,		ss.str(), paths.at("gxmem"), 		mm_size_bytes_C1, 	mm_Image_size, 		CV_32FC1, 		false /*true*/, 1);
																																			DownloadAndSaveVolume(		cdatabuf, 	ss.str(), paths.at("cdatabuf"), 	mm_size_bytes_C1,	mm_Image_size, 		CV_32FC1,  		false  , 1 /*max_range*/);
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
																																		if(verbosity>1) {
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
	
																																		if(verbosity>0) cout << "RunCL::allocatemem_finished\n\n" << flush;
}


void RunCL::predictFrame(){ //predictFrame();


}

void RunCL::loadFrame(cv::Mat image){ //getFrame();
																															if(verbosity>0) {cout << "\n RunCL::loadFrame_chk 0\n" << flush;}
	cl_int status;
	cl_event writeEvt;																										// WriteBuffer basemem #########
	status = clEnqueueWriteBuffer(uload_queue, basemem, CL_FALSE, 0, image_size_bytes, image.data, 0, NULL, &writeEvt);		if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
																															if (verbosity>1){
																																stringstream ss;	ss << frame_num;
																																DownloadAndSave_3Channel(basemem, ss.str(), paths.at("basemem"), image_size_bytes, baseImage_size,  baseImage_type, 	false );
																															}
}

void RunCL::cvt_color_space(){ //getFrame(); basemem(CV_8UC3, RGB)->imgmem(CV16FC3, HSV), NB we will use basemem for image upload, and imgmem for the MipMap. RGB is default for .png standard.
																															if(verbosity>1) {
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
	cl_event ev;
	res = clSetKernelArg(cvt_color_space_kernel, 0, sizeof(cl_mem), &basemem);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uchar3*		base,			//0
	res = clSetKernelArg(cvt_color_space_kernel, 1, sizeof(cl_mem), &imgmem);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global half*		img,			//1	
	res = clSetKernelArg(cvt_color_space_kernel, 2, sizeof(cl_mem), &uint_param_buf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uint*		uint_params		//2
	status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
																															if(verbosity>0) cout<<"\nRunCL::cvt_color_space()_chk1,  global_work_size="<< global_work_size <<flush;
	res = clEnqueueNDRangeKernel(m_queue, cvt_color_space_kernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, &ev); 							// run cvt_color_space_kernel  aka cvt_color_space(..) ##### TODO which CommandQueue to use ? What events to check ?
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status = clFlush(m_queue);				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents (1, &ev);		if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																															if(verbosity>1) cout<<"\nRunCL::cvt_color_space()_chk2"<<flush;
																															if (verbosity>1){
																																//cout<<",chk2.1,"<<flush;
																																stringstream ss;	ss << frame_num << "_cvt_color_space";
																																/*
																																cout<<",chk2.2,"<<flush;
																																cout<<"\n mm_size_bytes_C3="<< mm_size_bytes_C3 <<flush;
																																cout<<"\n mm_Image_size="<< mm_Image_size <<"\n\n"<<flush;
																																*/
																																DownloadAndSave_3Channel(	imgmem, ss.str(), paths.at("imgmem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4 /*mm_Image_type*/, 	false );
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
																															if(verbosity>1) cout<<"\nRunCL::cvt_color_space()_chk3_Finished"<<flush;
}
/*
#define MiM_PIXELS			0
#define MiM_READ_OFFSET		1
#define MiM_WRITE_OFFSET	2
#define MiM_READ_COLS		3
#define MiM_WRITE_COLS		4
#define MiM_GAUSSIAN_SIZE	5
*/
void RunCL::mipmap(uint num_reductions=4, uint gaussian_size=3){ //getFrame();
																																			if(verbosity>0) {cout<<"\n\nRunCL::mipmap(..)_chk0"<<flush;}
	cl_event 			writeEvt, ev;
	cl_int 				res, status;
	uint 				mipmap[8];
	float		a = float(0.0625);
	float		b = float(0.125);
	float		c = float(0.25);
	float 		gaussian[9] = {a, b, a, b, c, b, a , b, a };																																//  TODO load gaussian kernel & size from conf.json .
	if (gaussian_size!=3) {cout<<"Error: (gaussian_size!=3). Need to add code to malloc gaussian array. Probably with jsoncpp from 'conf.json' file." <<flush; exit(0); }
	
	status = clEnqueueWriteBuffer(uload_queue, gaussian_buf, CL_FALSE, 0, gaussian_size*gaussian_size*sizeof(float), gaussian, 0, NULL, &writeEvt);											// write mipmap_buf
	if (status != CL_SUCCESS){cout<<"\nstatus = "<<checkerror(status)<<"\n"<<flush; cout << "Error: RunCL::mipmap, clEnqueueWriteBuffer, mipmap_buf \n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	size_t local_size = local_work_size;																																							// set kernel args
	res = clSetKernelArg(mipmap_kernel, 0, sizeof(cl_mem), 					 	&imgmem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global half*	img,			//0	
	res = clSetKernelArg(mipmap_kernel, 1, sizeof(cl_mem), 					 	&gaussian_buf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global half*	gaussian,		//1
	res = clSetKernelArg(mipmap_kernel, 2, sizeof(cl_mem), 					 	&uint_param_buf);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global uint*	uint_params		//2
	res = clSetKernelArg(mipmap_kernel, 3, sizeof(cl_mem), 					 	&mipmap_buf);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global uint*	uint_params		//2
	res = clSetKernelArg(mipmap_kernel, 4, (local_size+2) * 4 * sizeof(cl_half4), NULL);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global uint*	uint_params		//2
	uint read_rows				= baseImage_size.height;
	uint write_rows 			= read_rows/2;
	mipmap[MiM_READ_OFFSET] 	= (mm_margin + (mm_margin * mm_width)); // 3* 
	mipmap[MiM_WRITE_OFFSET] 	= (baseImage_size.width + ( 3 * mm_margin) + (mm_margin * mm_width) ); // 3*   
	mipmap[MiM_READ_COLS] 		= baseImage_size.width;
	mipmap[MiM_WRITE_COLS] 		= mipmap[MiM_READ_COLS]/2;
	mipmap[MiM_GAUSSIAN_SIZE] 	= gaussian_size;
																																			if(verbosity>0) {cout<<"\n\nRunCL::mipmap(..)_chk1"<<flush;}
	for(int reduction = 0; reduction < num_reductions; reduction++) {
																																			if(verbosity>0) {cout<<"\n\nRunCL::mipmap(..)_chk2"<<flush;}
		mipmap[MiM_PIXELS]		= write_rows*mipmap[MiM_WRITE_COLS];																																	// compute num threads to launch & num_pixels in reduction
		size_t num_threads		= ceil( (float)(mipmap[MiM_PIXELS])/(float)local_work_size ) * local_work_size ;																						// global_work_size formula  
		
																																																		// write mipmap_buf
		status = clEnqueueWriteBuffer(uload_queue, mipmap_buf, 	CL_FALSE, 0, 8 * sizeof(uint), 	mipmap, 0, NULL, &writeEvt);	
		if (status != CL_SUCCESS){cout<<"\nstatus = "<<checkerror(status)<<"\n"<<flush; cout << "Error: RunCL::mipmap, clEnqueueWriteBuffer, mipmap_buf \n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
		res = clSetKernelArg(mipmap_kernel, 3, sizeof(cl_mem), &mipmap_buf);	
											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;																			//__global uint*	mipmap_params	//3
		status = clFlush(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}									// clEnqueueNDRangeKernel
		status = clFinish(m_queue); 		if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
		res = clEnqueueNDRangeKernel(m_queue, mipmap_kernel, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev); 																						// run mipmap_kernel, NB wait for own previous iteration.
		if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		status = clFlush(m_queue);			if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clWaitForEvents (1, &ev);	if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}					// update read&write rows&cols
		read_rows					= write_rows;
		write_rows 					= write_rows/2;
		mipmap[MiM_READ_OFFSET] 	= mipmap[MiM_WRITE_OFFSET];
		mipmap[MiM_WRITE_OFFSET] 	= mipmap[MiM_WRITE_OFFSET] + ( (read_rows + 2*mm_margin) * mm_width) ;  // 3*
		mipmap[MiM_READ_COLS] 		= mipmap[MiM_WRITE_COLS];
		mipmap[MiM_WRITE_COLS] 		= mipmap[MiM_WRITE_COLS]/2;
		mipmap[MiM_PIXELS] 			= mipmap[MiM_WRITE_COLS] * write_rows;
																																			if(verbosity>0) {cout<<"\n\nRunCL::mipmap(..)_chk2.6 Finished one loop"<<flush;}
	}
																																			if(verbosity>0) {
																																				cout<<"\n\nRunCL::mipmap(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << frame_num << "_mipmap";
																																				DownloadAndSave_3Channel(	imgmem, ss.str(), paths.at("imgmem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																			}
}

void RunCL::img_gradients(){ //getFrame();
	cl_event 			writeEvt, ev;
	cl_int 				res, status;
	size_t num_threads = mm_layerstep;
	
	res = clSetKernelArg(img_grad_kernel, 0, sizeof(cl_mem), &imgmem);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global half*	img,			//0	
	res = clSetKernelArg(img_grad_kernel, 1, sizeof(cl_mem), &uint_param_buf);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global uint*	uint_params		//2
	res = clSetKernelArg(img_grad_kernel, 2, sizeof(cl_mem), &gxmem);		 	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(img_grad_kernel, 3, sizeof(cl_mem), &gymem);		 	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(img_grad_kernel, 4, sizeof(cl_mem), &g1mem);		 	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	
	res = clEnqueueNDRangeKernel(m_queue, img_grad_kernel, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev); 																						// run mipmap_kernel, NB wait for own previous iteration.
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status = clFlush(m_queue);			if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents (1, &ev);	if (status != CL_SUCCESS)	{ cout << "\nclWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}					// update read&write rows&cols
		
}



void RunCL::loadFrameData(){ //getFrameData();


}

void RunCL::estimateSO3(){ //estimateSO3();


}

void RunCL::estimateSE3(){ //estimateSE3(); 				// own thread ? num iter ?


}

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

RunCL::~RunCL()
{
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
}

void RunCL::CleanUp()
{
																																			cout<<"\nRunCL::CleanUp_chk0"<<flush;
	cl_int status;
	status = clReleaseMemObject(basemem);	if (status != CL_SUCCESS)	{ cout << "\nbasemem  status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.1"<<flush;
	status = clReleaseMemObject(imgmem);	if (status != CL_SUCCESS)	{ cout << "\nimgmem   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.2"<<flush;
	status = clReleaseMemObject(cdatabuf);	if (status != CL_SUCCESS)	{ cout << "\ncdatabuf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.3"<<flush;
	status = clReleaseMemObject(hdatabuf);	if (status != CL_SUCCESS)	{ cout << "\nhdatabuf status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.4"<<flush;
	status = clReleaseMemObject(k2kbuf);	if (status != CL_SUCCESS)	{ cout << "\nk2kbuf   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.5"<<flush;
	status = clReleaseMemObject(qmem);		if (status != CL_SUCCESS)	{ cout << "\ndmem     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.7"<<flush;
	status = clReleaseMemObject(dmem);		if (status != CL_SUCCESS)	{ cout << "\ndmem     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.8"<<flush;
	status = clReleaseMemObject(amem);		if (status != CL_SUCCESS)	{ cout << "\namem     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.9"<<flush;
	status = clReleaseMemObject(lomem);		if (status != CL_SUCCESS)	{ cout << "\nlomem    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.10"<<flush;
	status = clReleaseMemObject(himem);		if (status != CL_SUCCESS)	{ cout << "\nhimem    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>0) cout<<"\nRunCL::CleanUp_chk0.11"<<flush;
																																			cout<<"\nRunCL::CleanUp_chk1_finished"<<flush;
}

void RunCL::exit_(cl_int res)
{
	CleanUp();
	//~RunCL(); Never need to call a destructor manually.
	exit(res);
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
    
    
