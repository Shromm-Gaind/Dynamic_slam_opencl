#include "Dynamic_slam.h"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::initialize_keyframe_from_GT(){																							// GT depth map is for current GT pose.
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::initialize_keyframe_from_GT()_chk 0" << flush;}
	keyframe_pose_GT 		= pose_GT;
	keyframe_inv_pose_GT 	= getInvPose(keyframe_pose_GT);
	keyframe_inv_K_GT		= generate_invK_(K_GT);

	keyframe_pose 			= pose_GT;
	keyframe_K				= K_GT;
	keyframe_inv_pose 		= inv_pose_GT;
	keyframe_inv_K			= inv_K_GT;

	keyframe_old_K			= old_K_GT;
	keyframe_old_pose		= old_pose_GT;

	keyframe_K2K 			= K2K_GT;						// TODO chk wrt when this is called and what values it would hold.
	keyframe_pose2pose 		= pose2pose_GT;

	runcl.initializeDepthCostVol( runcl.depth_mem_GT );
	initialize_new_keyframe();
}

void Dynamic_slam::initialize_keyframe_from_tracking(){																						// NB need to transform depth map from previous keyfrae to current pose.
	int local_verbosity_threshold = -1;																										if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::initialize_keyframe_from_tracking()_chk 0" << flush;}
	keyframe_old_pose		= keyframe_pose;
	keyframe_old_K			= keyframe_K;
	keyframe_pose 			= pose;
	keyframe_K				= K;
	keyframe_inv_pose 		= inv_pose;
	keyframe_inv_K			= inv_K;

	cv::Matx44f inv_pose2pose = getInvPose(keyframe_pose2pose);																				//cv::Matx44f Dynamic_slam::getInvPose(cv::Matx44f pose)
	cv::Matx44f forward_keyframe2K  = K * inv_pose2pose * inv_old_K;
																																			if(verbosity>local_verbosity_threshold){
																																				PRINT_MATX44F(K,);
																																				PRINT_MATX44F(keyframe_pose2pose,);
																																				PRINT_MATX44F(inv_pose2pose,);
																																				PRINT_MATX44F(inv_old_K,);
																																				PRINT_MATX44F(forward_keyframe2K,);
																																			}
	runcl.transform_depthmap(forward_keyframe2K, runcl.keyframe_depth_mem );																// NB runcl.transform_depthmap(..) must be used _before_ initializing the new cost_volume, because it uses keyframe_basemem.
	runcl.initializeDepthCostVol( runcl.depth_mem );
	initialize_new_keyframe();
}

void Dynamic_slam::initialize_new_keyframe(){
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::initialize_new_keyframe()_chk 0,  runcl.dataset_frame_num = "<< runcl.dataset_frame_num << flush;}
	runcl.initialize_fp32_params();
	//runcl.QD_count 	= 0; // TODO NB these are reset in Dynamic_slam::nextFrame()
	//runcl.A_count 	= 0;
	//runcl.G_count 	= 0;

	//cacheGValues();			// TODO may not be needed here.
								// TODO   keyframe_K2K_GT, keyframe_K2K etc ?
	runcl.keyFrameCount++;
	runcl.dataset_frame_num++;
}
