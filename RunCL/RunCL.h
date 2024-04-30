#ifndef RUNCL_H
#define RUNCL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION		200
#define CL_TARGET_OPENCL_VERSION			200  // defined as 120, ie OpenCL 1.2 in CMakeLists.txt
#define CL_HPP_TARGET_OPENCL_VERSION		200  // OpenCL 2.0

#include <CL/opencl.hpp>
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

#include "../utils/conf_params.hpp"
#include "../utils/convertTransforms.hpp"
#include "../utils/print_functions.hpp"
#include "../kernels/kernels_macros.h"

const uint tracking_num_colour_channels = 4;

using namespace std;
class RunCL
{
public:
	//RunCL( conf_params j_params ); // map<string, Json::Value> obj_
	RunCL( Json::Value obj_ , int_map verbosity_mp );
	Json::Value 		obj;
	int_map 			verbosity_mp;

	cv::Mat 			resultsMat;						// used to insert images for multiple iterations, and variables for comparison. Size set in itialization, from cnf.json data.
	int					verbosity;
	bool				tiff, png;
	std::vector<cl_platform_id> 	m_platform_ids;
	cl_context			m_context;
	cl_device_id		m_device_id;
	cl_command_queue	m_queue, uload_queue, dload_queue, track_queue;
	cl_program			m_program;
	cl_kernel			convert_depth_kernel, invert_depth_kernel, transform_depthmap_kernel, depth_cost_vol_kernel, cost_kernel, cache3_kernel, cache4_kernel, updateQD_kernel, updateG_kernel, updateA_kernel, measureDepthFit_kernel;
	cl_kernel			cvt_color_space_kernel, cvt_color_space_linear_kernel, img_variance_kernel, blur_image_kernel, reduce_kernel, mipmap_float4_kernel, mipmap_float_kernel, img_grad_kernel, se3_rho_sq_kernel, comp_param_maps_kernel;
	cl_kernel			se3_lk_grad_kernel, atomic_test1_kernel;
	
	//bool 				frame_bool_idx=0;
	cl_mem 				basemem, imgmem,  imgmem_blurred, gxmem, gymem, g1mem,  k_map_mem, dist_map_mem, SE3_grad_map_mem, SE3_incr_map_mem;
	cl_mem				cdatabuf, cdatabuf_8chan, hdatabuf, dmem, amem, qmem, qmem2, lomem, himem, img_sum_buf, depth_mem, depth_mem_GT;											// NB 'depth_mem' is that used by tracking & auto-calibration.
	cl_mem				k2kbuf, SO3_k2kbuf, SE3_k2kbuf, fp32_param_buf, uint_param_buf, mipmap_buf, gaussian_buf, img_stats_buf, SE3_map_mem, SE3_rho_map_mem, se3_sum_rho_sq_mem, SE3_weight_map_mem;	// param_map_mem,
	cl_mem 				pix_sum_mem, var_sum_mem, se3_sum_mem, se3_sum2_mem, se3_weight_sum_mem;																										// reduce_param_buf;
	cl_mem 				keyframe_imgmem, keyframe_imgmem_HSV_grad, keyframe_depth_mem, keyframe_g1mem, keyframe_SE3_grad_map_mem, keyframe_depth_mem_GT;							// keyframe_gxmem, keyframe_gymem, keyframe_basemem,
	cl_mem				HSV_grad_mem, dmem_disparity, dmem_disparity_sum;
	cl_mem				atomic_test1_buf;
	
	cv::Mat 			baseImage;
	size_t  			global_work_size, mm_global_work_size, local_work_size, image_size_bytes, image_size_bytes_C1, mm_size_bytes_C1, mm_size_bytes_C3, mm_size_bytes_C4, mm_size_bytes_C8, mm_size_bytes_half4, mm_vol_size_bytes;
	size_t 				so3_sum_size, so3_sum_size_bytes, mm_se3_sum_size, se3_sum_size, se3_sum_size_bytes, se3_sum2_size_bytes, pix_sum_size, pix_sum_size_bytes;
	size_t 				d_disp_sum_size, d_disp_sum_size_bytes;
	bool 				gpu, amdPlatform;
	cl_device_id 		deviceId;
	
	size_t				img_stats_size_bytes = sizeof(float)*8*4*2;
	float				img_stats[8*4*2]	= {0};		// 8 layers, 4 channels, 2 variables.
	size_t 				num_threads[8]		= {0};
	uint 				MipMap[8*8]			= {0};
	uint				uint_params[8]		= {0};
	
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

	///////////////////////////////////// RunCL_class.cpp


	void testOpencl();
	void getDeviceInfoOpencl(cl_platform_id platform);
	int  convertToString(const char *filename, std::string& s);
	void createQueues();
	void createAndBulidProgramFromSource(cl_device_id *devices);
	void createKernels();

	int  waitForEventAndRelease(cl_event *event);
	void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential, const size_t local_work_size);						// Call kernels on mipmap: start,stop allow running specific layers.

	void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential=false){ mipmap_call_kernel( kernel_to_call,  queue_to_call, mm_start, mm_stop, layers_sequential, local_work_size); }

	void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call){ mipmap_call_kernel( kernel_to_call,  queue_to_call, mm_start, mm_stop, false, local_work_size); } // , true

	// 	mipmap_call_kernel( 	depth_cost_vol_kernel, 		m_queue, 	start, 	stop );

	void initialize_fp32_params();
	void initialize_RunCL();																													// Setting up buffers & mipmap parameters
	void allocatemem();

	void CleanUp();																														// Exit...
	void exit_(cl_int res);
	~RunCL();

	/////////////////////////////////////// RunCL_macro_conversion.cpp

	string 	checkerror(int input);
	string 	checkCVtype(int input);

	/////////////////////////////////////// RunCL_DownloadAndSave.cpp

	void createFolders();																												// Called by RunCL(..) constructor, above.
	void ReadOutput(uchar* outmat) ;
	void ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0) ;
	void saveCostVols(float max_range);

	void DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1 );
	void DownloadAndSave_2Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1, uint offset=0, bool exception_tiff=false ){
		cv::Mat bufImg;
		DownloadAndSave_3Channel(buffer, count, folder_tiff, image_size_bytes,  size_mat,  type_mat,  show,  &bufImg,  max_range, offset, exception_tiff );
	}
	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range=1, uint offset=0, bool exception_tiff=false );
	void DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers,  bool exception_tiff=false, float iter=0, bool display=false );

	void PrepareResults_3Channel(cl_mem buffer, size_t image_size_bytes, cv::Size size_mat, int type_mat, cv::Mat *bufImg, float max_range /*=1*/, uint offset /*=0*/ );
	void PrepareResults_3Channel_volume(cl_mem buffer, size_t image_size_bytes, cv::Size size_mat, int type_mat, float max_range, uint vol_layers,  float iter);

	void DownloadAndSave_6Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0);
	void DownloadAndSave_6Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_8Channel(cl_mem buffer, std::string count, std::map< std::string, boost::filesystem::path > folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/);
	void DownloadAndSave_8Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_HSV_grad(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0 );
	
	void SaveMat(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	void SaveMat_1chan(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	void DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, bool exception_tiff=false );
/*
	void DownloadAndSave_buffer (																										// Shelve linearMipMap for now. It will not make kernels simpler or faster.
		cl_mem 		buffer,
		std::string count,
		std::map< std::string, boost::filesystem::path > folder_tiff,
		size_t      image_size_bytes,
		cv::Size    size_mat,
		int 		type_mat_out,		// (data size and channels)
		int 		num_channels_out,
		int 		num_channels_in,
		int 		maps_in_vol,
		int 		start_layer,
		int 		stop_layer,
		float 		max_range,
		bool 		exception_tiff,
		bool 		show
	);
*/
	////////////////////////////////////// RunCL_load_image.cpp

	void precom_param_maps(float SO3_k2k[6*16]);																						// Image loading & preparation
	void loadFrame(cv::Mat image);
	void cvt_color_space();
	void img_variance();
	void blur_image();
	void mipmap_linear();
	void img_gradients();
	
	void load_GT_depth(cv::Mat GT_depth, bool invert);																					// Depthmap loading & preparation
	void convert_depth(uint invert, float factor);
	void mipmap_depthmap(cl_mem depthmap_);
	
	/////////////////////////////////////// RunCL_tracking.cpp
	
	void se3_rho_sq(float Rho_sq_results[8][4], const float count[4], uint start, uint stop);							 				// Tracking

	void estimateSE3_LK(float SE3_results[8][6][tracking_num_colour_channels], float SE3_weights_results[8][6][tracking_num_colour_channels], float Rho_sq_results[8][4], int count, uint start, uint stop);

	void read_Rho_sq(float Rho_sq_results[8][4]);
	void read_se3_weights(float SE3_weights_results[8][6][tracking_num_colour_channels]);
	void read_se3_incr(float SE3_results[8][6][tracking_num_colour_channels]);

	void writeToResultsMat(cv::Mat *bufImg  , uint column_of_images , uint row_of_images );
	void tracking_result(string result);
	void estimateCalibration();																											// Camera calibration
	void RelativeVel_Map();																												// RelativeVelMap - placeholder...
	void atomic_test1();

	/////////////////////////////////////// RunCL_mapping.cpp

	void transform_depthmap(cv::Matx44f K2K_ , cl_mem depthmap_);																		// Cost volume
	void initializeDepthCostVol(cl_mem key_frame_depth_map_src);	// Depth costvol functions
	void updateDepthCostVol(cv::Matx44f K2K_, int count, uint start, uint stop);
	void updateQD(float epsilon, float theta, float sigma_q, float sigma_d, uint start, uint stop);
	void updateG(int count, uint start, uint stop);
	void updateA(float lambda, float theta, uint start, uint stop);
	void computeSigmas(float epsilon, float theta, float L, float &sigma_d, float &sigma_q);
	
	void measureDepthFit(uint start, uint stop);

	void SpatialCostFns();																												// SIRFS cost functions
	void ParsimonyCostFns();
	void ExhaustiveSearch();
};
#endif /*RUNCL_H*/
