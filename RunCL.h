#ifndef RUNCL_H
#define RUNCL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION		200
#define CL_TARGET_OPENCL_VERSION			200  // defined as 120 in CMakeLists.txt 
#define CL_HPP_TARGET_OPENCL_VERSION		200  // OpenCL 1.2

#include <CL/opencl.hpp>	//<CL/cl.hpp>
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

//#define PIXELS			0	// fp16_params indices
//#define ROWS				1	// TODO Can these be #included from a common header for both host and device code?
//#define COLS				2
//#define LAYERS				3
#define MAX_INV_DEPTH		0	// fp16_params indices
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

#define PIXELS				0	// uint_params indices
#define ROWS				1
#define COLS				2
#define LAYERS				3
#define MARGIN				4
#define MM_PIXELS			5
#define MM_ROWS				6
#define MM_COLS				7

#define MiM_PIXELS			0	// MipMap uint params
#define MiM_READ_OFFSET		1
#define MiM_WRITE_OFFSET	2
#define MiM_READ_COLS		3
#define MiM_WRITE_COLS		4
#define MiM_GAUSSIAN_SIZE	5

/*
#define BASE_MEM 			0	// device memory buffers
#define IMG_MEM				1
#define CDATA_BUF			2
#define HDATA_BUF			3
#define K2K_BUF				4
#define D_MEM				5
#define A_MEM				6
#define BASEGRAY_MEM		7
#define GX_MEM				8
#define GY_MEM				9
#define LO_MEM				10
#define HI_MEM				11
#define PARAM_BUF			12
#define IMG_SUM_BUF			13
#define NUM_MEM_BUFS		13

#define COST_KERNEL			0	// kernels
#define CACHE_KERNEL		1
#define UPDATE_QD_KERNEL	2
#define UPDATE_A_KERNEL		3
#define NUM_KERNELS			3

#define M_QUEUE				0	// comman queues
#define ULOAD_QUEUE			1
#define DLOAD_QUEUE			2
#define TRACK_QUEUE			3
#define NUM_COMMAND_QUEUES	3
*/
using namespace std;
class RunCL
{
public:
	Json::Value 		obj;
	int					verbosity;
	
	std::vector<cl_platform_id> 	m_platform_ids;
	cl_context						m_context;
	cl_device_id					m_device_id;
	cl_command_queue				m_queue, uload_queue, dload_queue, track_queue; // queue[4]; //
	cl_program						m_program;
	cl_kernel						cost_kernel, cache3_kernel, cache4_kernel, updateQD_kernel, updateA_kernel; // kern[4]; //
	cl_kernel						cvt_color_space_kernel, mipmap_kernel, img_grad_kernel;
	cl_mem							basemem, imgmem, cdatabuf, hdatabuf, dmem, amem, basegraymem, gxmem, gymem, g1mem, qmem, lomem, himem, img_sum_buf; // mem[14]; //
	cl_mem							k2kbuf, half_param_buf, fp16_param_buf, uint_param_buf, mipmap_buf, gaussian_buf;
	
	//cl_event						;
	
	
	cv::Mat 			baseImage;
	size_t  			global_work_size, mm_global_work_size, local_work_size, image_size_bytes, mm_size_bytes_C1, mm_size_bytes_C3, mm_vol_size_bytes;
	bool 				gpu, amdPlatform;
	cl_device_id 		deviceId;
	
	uint				uint_params[8] 		= {0};
	cv::float16_t 		fp16_params[16]		= { cv::float16_t(0) };
	cv::float16_t 		fp16_k2k[16]		= { cv::float16_t(0) };
	//cl_half				cl_half_params[16]		= { cl_half(0) };
	//cl_half				cl_half_k2k[16]			= { cl_half(0) };
	
	int 				frame_num;
	uint 				mm_margin, mm_height, mm_width, mm_layerstep, fp16_size; 
	int 				width, height, layerstep, costVolLayers, baseImage_type, mm_Image_type, count=0, keyFrameCount=0, costVolCount=0, QDcount=0, A_count=0;
	cv::Size 			baseImage_size, mm_Image_size;
	std::map< std::string, boost::filesystem::path > paths;

	RunCL(Json::Value obj_);
	void createFolders();	// Called by RunCL(..) constructor, above.
	void saveCostVols(float max_range);
	void DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );
	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show );
	void DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );
	
	//void computeSigmas(float epsilon, float theta, float L, cv::float16_t &sigma_d, cv::float16_t &sigma_q );
	void computeSigmas(float epsilon, float theta, float L, cv::float16_t &sigma_d, cv::float16_t &sigma_q);
	void computeSigmas(float epsilon, float theta, float L, cl_half       &sigma_d, cl_half       &sigma_q);

	void allocatemem();//float* gx, float* gy, float* params, int layers, cv::Mat &baseImage, float *cdata, float *hdata, float *img_sum_data);
	void calcCostVol(float* k2k, cv::Mat &image);
	void cacheGValue2(cv::Mat &bgray, float theta);
	void updateQD(float epsilon, float theta, float sigma_q, float sigma_d);
	void updateA ( float lambda, float theta );

	///
	void predictFrame();
	void loadFrame(cv::Mat image);
	void cvt_color_space();
	void mipmap(uint num_reductions, uint gaussian_size);
	void img_gradients();
	
	void loadFrameData();
	void estimateSO3();
	void estimateSE3();
	void estimateCalibration();
	void buildDepthCostVol();
	void SpatialCostFns();
	void ParsimonyCostFns();
	void ExhaustiveSearch();
	///
	void CleanUp();
	void exit_(cl_int res);
	~RunCL();

	//void DownloadAndSaveVolume_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show );
	//void initializeCostVol(float* k2k, cv::Mat &baseImage, cv::Mat &image, float *cdata, float *hdata, float thresh, int layers);
	//void initializeAD();

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
		ReadOutput(outmat, amem,  (width * height * sizeof(float)) );
	}

	void ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0) {
		cl_event readEvt;
		cl_int status;
														//cout<<"\nReadOutput: &outmat="<<&outmat<<", buf_mem="<<buf_mem<<", data_size="<<data_size<<", offset="<<offset<<"\n"<<flush;
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
															//(else if(verbosity>0) cout <<"\nclFlush(..)"<<flush;
		status = clWaitForEvents(1, &readEvt); 			if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status="			<< checkerror(status) <<"\n"<<flush; exit_(status);} 
															//else if(verbosity>0) cout <<"\nclWaitForEvents(..)"<<flush;
		//status = clFinish(dload_queue);					if (status != CL_SUCCESS) { cout << "\nclFinish(m_queue) status = " 		<< checkerror(status) <<"\n"<<flush; exit_(status);} 
															//else if(verbosity>0) cout <<"\nclFlush(..)"<<flush;
	}
};

#endif /*RUNCL_H*/
