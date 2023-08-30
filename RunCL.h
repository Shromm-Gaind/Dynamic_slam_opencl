#ifndef RUNCL_H
#define RUNCL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION		200
#define CL_TARGET_OPENCL_VERSION			200  // defined as 120 in CMakeLists.txt 
#define CL_HPP_TARGET_OPENCL_VERSION		200  // OpenCL 1.2

#include <CL/opencl.hpp>		//<CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <jsoncpp/json/json.h>
#include "utils/utils.hpp"

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

using namespace std;
class RunCL
{
public:
	Json::Value 		obj;
	int					verbosity;
	bool				tiff, png;
	std::vector<cl_platform_id> 	m_platform_ids;
	cl_context			m_context;
	cl_device_id		m_device_id;
	cl_command_queue	m_queue, uload_queue, dload_queue, track_queue;
	cl_program			m_program;
	cl_kernel			convert_depth_kernel, invert_depth_kernel, transform_depthmap_kernel, depth_cost_vol_kernel, cost_kernel, cache3_kernel, cache4_kernel, updateQD_kernel, updateG_kernel, updateA_kernel;
	cl_kernel			cvt_color_space_kernel, cvt_color_space_linear_kernel, img_variance_kernel, reduce_kernel, mipmap_float4_kernel, mipmap_float_kernel, img_grad_kernel, so3_grad_kernel, se3_grad_kernel, comp_param_maps_kernel;
	
	//bool 				frame_bool_idx=0;
	cl_mem 				basemem, imgmem,  gxmem, gymem, g1mem,  k_map_mem, dist_map_mem, SE3_grad_map_mem, SE3_incr_map_mem;
	cl_mem				cdatabuf, hdatabuf, dmem, amem, qmem, qmem2, lomem, himem, img_sum_buf, depth_mem, depth_mem_GT;											// NB 'depth_mem' is that used by tracking & auto-calibration.
	cl_mem				k2kbuf, SO3_k2kbuf, SE3_k2kbuf, fp32_param_buf, uint_param_buf, mipmap_buf, gaussian_buf, img_stats_buf, SE3_map_mem, SE3_rho_map_mem, se3_sum_rho_sq_mem;	// param_map_mem,  
	cl_mem 				pix_sum_mem, var_sum_mem, se3_sum_mem, se3_sum2_mem;					// reduce_param_buf;
	cl_mem 				keyframe_imgmem, keyframe_depth_mem, keyframe_g1mem, keyframe_SE3_grad_map_mem;	// keyframe_gxmem, keyframe_gymem, keyframe_basemem, 
	cl_mem				HSV_grad_mem;
	
	cv::Mat 			baseImage;
	size_t  			global_work_size, mm_global_work_size, local_work_size, image_size_bytes, image_size_bytes_C1, mm_size_bytes_C1, mm_size_bytes_C3, mm_size_bytes_C4, mm_size_bytes_half4, mm_vol_size_bytes;
	size_t 				so3_sum_size, so3_sum_size_bytes, mm_se3_sum_size, se3_sum_size, se3_sum_size_bytes, se3_sum2_size_bytes, pix_sum_size, pix_sum_size_bytes;
	bool 				gpu, amdPlatform;
	cl_device_id 		deviceId;
	
	size_t				img_stats_size_bytes = sizeof(float)*8*4*2;
	float				img_stats[8*4*2]	= {0};		// 8 layers, 4 channels, 2 variables.
	size_t 				num_threads[8]		= {0};
	uint 				MipMap[8*8]			= {0};
	uint				uint_params[8] 		= {0};
	
	float				fp32_params[16]		= {0};
	float				fp32_so3_k2k[9]		= {0};
	float				fp32_k2k[16]		= {0};
	float 				fp32_k2keyframe[16]	= {0};
	
	uint	 			mm_num_reductions;				//	
	int 				mm_gaussian_size;				//	
	int 				mm_margin;						//	
	int 				mm_height;						//	
	int 				mm_width;						//	
	int 				mm_layerstep;					//	
	int 				fp16_size;
	uint 				mm_start;						//	
	int 				mm_stop;
	int 				baseImage_width;				//	
	int 				baseImage_height;				//	
	int 				layerstep;						//	
	int 				costVolLayers;					//	
	int 				baseImage_type;					//	
	int 				mm_Image_type;					//	
	
	int 				dataset_frame_num;				//	Frame number in dataset, set in constructor from json file. Incremented in Dynamic_slam::nextFrame.
	int 				costvol_frame_num;				//	Frame number in the cost volume. Set = 0 in RunCL::initializeDepthCostVol(..) . Incremented in Dynamic_slam::nextFrame(..)
	int 				keyFrameCount			= 0;	//	used in saving data to file. Incremented in Dynamic_slam::initialize_new_keyframe(..)
	int 				save_index				= 0;	//	Set in RunCL::initializeDepthCostVol(), and RunCL::updateDepthCostVol(), to save_index = keyFrameCount*1000 + costvol_frame_num;
	
	int 				QD_count 				= 0; 	//	Incremented in RunCL::updateQD(..) Set = 0 in Dynmaic_slam::initialize_new_keyfrme(..)  & in Dynamic_slam::nextFrame()
	int 				A_count					= 0;	//	Incremented in RunCL::updateA(..), ditto
	int 				G_count					= 0;	//	Incremented in RunCL::updateG(..), ditto
	
	cv::Size 			baseImage_size, mm_Image_size;
	std::map< std::string, boost::filesystem::path > paths;

	RunCL(Json::Value obj_);
	void testOpencl();
	void getDeviceInfoOpencl(cl_platform_id platform);
	void createFolders();																												// Called by RunCL(..) constructor, above.
	
	void saveCostVols(float max_range);
	void DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );
	void DownloadAndSave_2Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1, uint offset=0 );
	void DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	void DownloadAndSave_3Channel_linear_Mipmap(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, int type_mat, bool show );
	
	void DownloadAndSave_6Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset);
	void DownloadAndSave_6Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_HSV_grad(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset );
	
	void SaveMat(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	void DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );
	
	void initialize_fp32_params();
	void initialize();																													// Setting up buffers & mipmap parameters
	void allocatemem();
	void precom_param_maps(float SO3_k2k[6*16]);
	
	void predictFrame();																												// Image loading & preparation
	void loadFrame(cv::Mat image);
	void cvt_color_space();
	void img_variance();
	void mipmap_linear();
	void img_gradients();
	
	void load_GT_depth(cv::Mat GT_depth, bool invert);																					// Depthmap loading & preparation
	void convert_depth(uint invert, float factor);
	void mipmap_depthmap(cl_mem depthmap_);
	
	void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential=false);						// Call kernels on mipmap: start,stop allow running specific layers.
	void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call){
		mipmap_call_kernel( kernel_to_call,  queue_to_call, mm_start, mm_stop );
	}
	
	void estimateSO3(float SO3_results[8][3][4], float Rho_sq_results[8][4], int count, uint start, uint stop);   						// Tracking
	void estimateSE3(float SE3_results[8][6][4], float Rho_sq_results[8][4], int count, uint start, uint stop);
	void tracking_result(string result);
	
	void estimateCalibration();																											// Camera calibration
	
	void transform_depthmap(cv::Matx44f K2K_ , cl_mem depthmap_);																		// Cost volume
	void initializeDepthCostVol(cl_mem key_frame_depth_map_src);	// Depth costvol functions
	   void updateDepthCostVol(cv::Matx44f K2K_, int count, uint start, uint stop);
	void updateQD(float epsilon, float theta, float sigma_q, float sigma_d, uint start, uint stop);
	void updateG(int count, uint start, uint stop);
	void updateA(float lambda, float theta, uint start, uint stop);
	void computeSigmas(float epsilon, float theta, float L, float &sigma_d, float &sigma_q);

	void SpatialCostFns();																												// SIRFS cost functions
	void ParsimonyCostFns();
	void ExhaustiveSearch();
	
	void RelativeVelMap();																												// RelativeVelMap - placeholder...
	
	void CleanUp();																														// Exit...
	void exit_(cl_int res);
	~RunCL();

	/*
	//void computeSigmas(float epsilon, float theta, float L, cv::float16_t &sigma_d, cv::float16_t &sigma_q);
	//void computeSigmas(float epsilon, float theta, float L, cl_half &sigma_d, cl_half &sigma_q);
	//void mipmap(uint num_reductions, uint gaussian_size);
	//void loadFrameData();
	//void loadFrameData(cv::Mat GT_depth, cv::Matx44f GT_K2K,   cv::Matx44f GT_pose2pose);
	//void generate_SE3_k2k(float *_SE3_k2k);
	//void estimateSE3 ( float SE3_results[4][6][8], float Rho_sq_results[4][8], int count = 48, uint start = 0, uint stop = 8 );
	//void calcCostVol(float* k2k, cv::Mat &image);
	//void cacheGValue2(cv::Mat &bgray, float theta);
	//void DownloadAndSaveVolume_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show );
	//void initializeCostVol(float* k2k, cv::Mat &baseImage, cv::Mat &image, float *cdata, float *hdata, float thresh, int layers);
	//void initializeAD();
	*/

	int  convertToString(const char *filename, std::string& s){
		size_t size;
		char*  str;
		std::fstream f(filename, (std::fstream::in | std::fstream::binary));
		if (f.is_open() ) {
			size_t fileSize;
			f.seekg(0, std::fstream::end);
			size = fileSize = (size_t)f.tellg();
			f.seekg(0, std::fstream::beg);
			str = new char[size + 1];
			if (!str) {
				f.close();
				return 0;
			}
			f.read(str, fileSize);
			f.close();
			str[size] = '\0';
			s = str;
			delete[] str;
			return 0;
		}
											cout << "Error: failed to open file\n:" << filename << endl;
		return 1;
	}

	int waitForEventAndRelease(cl_event *event){
											if(verbosity>0) cout << "\nwaitForEventAndRelease_chk0, event="<<event<<" *event="<<*event << flush;
		cl_int status = CL_SUCCESS;
		status = clWaitForEvents(1, event); if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status=" << status << ", " <<  checkerror(status) <<"\n" << flush; exit_(status); }
		status = clReleaseEvent(*event); 	if (status != CL_SUCCESS) { cout << "\nclReleaseEvent status="  << status << ", " <<  checkerror(status) <<"\n" << flush; exit_(status); }
		return status;
	}

	string checkerror(int input) {
		int errorCode = input;
		switch (errorCode) {
		case -9999:											return "Illegal read or write to a buffer";		// NVidia error code
		case CL_DEVICE_NOT_FOUND:							return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:						return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:						return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:							return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:							return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:				return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:							return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:						return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:						return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:								return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_INVALID_VALUE:								return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:						return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:							return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:								return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:							return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:					return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:						return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:							return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:							return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:							return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:							return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:								return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:						return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:							return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:					return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:						return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:					return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:								return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:							return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:							return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:							return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:						return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:						return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:					return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:						return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:						return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:					return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:								return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:							return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:							return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:						return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:							return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:					return "CL_INVALID_GLOBAL_WORK_SIZE";
#if CL_HPP_MINIMUM_OPENCL_VERSION >= 200 
		case CL_INVALID_DEVICE_QUEUE:						return "CL_INVALID_DEVICE_QUEUE";
		case CL_INVALID_PIPE_SIZE:							return "CL_INVALID_PIPE_SIZE";
#endif
		default:											return "unknown error code";
		}
	}

	string checkCVtype(int input) {
		int errorCode = input;
		switch (errorCode) {
			//case	CV_8U:										return "CV_8U";
			case	CV_8UC1:										return "CV_8UC1";
			case	CV_8UC2:										return "CV_8UC2";
			case	CV_8UC3:										return "CV_8UC3";
			case	CV_8UC4:										return "CV_8UC4";

			//case	CV_8S:										return "CV_8S";
			case	CV_8SC1:										return "CV_8SC1";
			case	CV_8SC2:										return "CV_8SC2";
			case	CV_8SC3:										return "CV_8SC3";
			case	CV_8SC4:										return "CV_8SC4";

			//case	CV_16U:										return "CV_16U";
			case	CV_16UC1:										return "CV_16UC1";
			case	CV_16UC2:										return "CV_16UC2";
			case	CV_16UC3:										return "CV_16UC3";
			case	CV_16UC4:										return "CV_16UC4";

			//case	CV_16S:										return "CV_16S";
			case	CV_16SC1:										return "CV_16SC1";
			case	CV_16SC2:										return "CV_16SC2";
			case	CV_16SC3:										return "CV_16SC3";
			case	CV_16SC4:										return "CV_16SC4";

			//case	CV_32S:										return "CV_32S";
			case	CV_32SC1:										return "CV_32SC1";
			case	CV_32SC2:										return "CV_32SC2";
			case	CV_32SC3:										return "CV_32SC3";
			case	CV_32SC4:										return "CV_32SC4";

			//case	CV_16F
			case	CV_16FC1:										return "CV_16FC1";
			case	CV_16FC2:										return "CV_16FC2";
			case	CV_16FC3:										return "CV_16FC3";
			case	CV_16FC4:										return "CV_16FC4";
			
			//case	CV_32F:										return "CV_32F";
			case	CV_32FC1:										return "CV_32FC1";
			case	CV_32FC2:										return "CV_32FC2";
			case	CV_32FC3:										return "CV_32FC3";
			case	CV_32FC4:										return "CV_32FC4";

			//case	CV_64F:										return "CV_64F";
			case	CV_64FC1:										return "CV_64FC1";
			case	CV_64FC2:										return "CV_64FC2";
			case	CV_64FC3:										return "CV_64FC3";
			case	CV_64FC4:										return "CV_64FC4";

			default:										return "unknown CV_type code";
		}
	}

	void ReadOutput(uchar* outmat) {
		ReadOutput(outmat, amem,  (baseImage_width * baseImage_height * sizeof(float)) );
	}

	void ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0) {
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
};

#endif /*RUNCL_H*/
