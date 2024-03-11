#pragma once

#include <string>
#include <boost/filesystem.hpp>
#include <fstream>
#include <set>
#include "utils/convertAhandaPovRayToStandard.h"
#include "RunCL/RunCL.h"

#define BOOST_FILESYSTEM_VERSION          3
#define BOOST_FILESYSTEM_NO_DEPRECATED 

#define Rx	0
#define Ry  1
#define Rz	2
#define Tx	3
#define Ty	4
#define Tz	5

namespace fs = ::boost::filesystem;

class Dynamic_slam
{
  public:
    ~Dynamic_slam();
    Dynamic_slam( Json::Value obj_ );
    
    Json::Value obj;
    RunCL runcl;
    int verbosity;
    bool invert_GT_depth = false;
    int SE_iter_per_layer, SE3_stop_layer, SE3_start_layer, SE_iter;
    float SE3_Rho_sq_threshold, SE_factor;
    
    // camera & pose params
    const cv::Matx44f Matx44f_zero = {0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0}; //  = cv::Matx44f::zeros();//
    const cv::Matx44f Matx44f_eye  = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};
    const cv::Matx33f Matx33f_zero = {0,0,0,  0,0,0,  0,0,0 };                //  = cv::Matx33f::zeros();//
    const cv::Matx33f Matx33f_eye  = {1,0,0,  0,1,0,  0,0,1 };
    
    cv::Matx33f SO3_pose2pose;
    cv::Matx31f SO3_pose2pose_algebra;
    
    cv::Matx44f K, inv_K, pose, inv_pose, K2K, pose2pose, old_K, inv_old_K, old_pose, inv_old_pose, transform[6]  ;
    cv::Matx61f pose2pose_algebra_0, pose2pose_algebra_1, pose2pose_algebra_2;
    
    cv::Matx44f K_GT, inv_K_GT, pose_GT, inv_pose_GT, K2K_GT, pose2pose_GT, old_K_GT, inv_old_K_GT, old_pose_GT, inv_old_pose_GT, transform_GT[6]  ; // Ground Truth from the dataset.
    cv::Matx44f K_start, inv_K_start, pose_start, inv_pose_start, K2K_start, pose2pose_start ;
    cv::Matx44f pose2pose_accumulated, pose2pose_accumulated_GT;
    
    cv::Matx61f pose2pose_accumulated_GT_algebra, pose2pose_accumulated_algebra, pose2pose_accumulated_error_algebra, pose2pose_GT_algebra, pose2pose_algebra, pose2pose_error_algebra;
    
    cv::Mat image, R, T, depth_GT, cameraMatrix, projection;       // TODO should these be Matx ? 
    cv::Mat old_R, old_T, R_dif, T_dif;
    float SE3_k2k[6*16];
    
    // lens distortion params
    
    
    // keyframe params
    cv::Matx44f keyframe_K, keyframe_inv_K, keyframe_pose, keyframe_inv_pose, keyframe_K2K, keyframe_pose2pose, keyframe_old_K, keyframe_inv_old_K, keyframe_old_pose, keyframe_inv_old_pose;
    cv::Matx44f keyframe_K_GT, keyframe_inv_K_GT, keyframe_pose_GT, keyframe_inv_pose_GT, keyframe_K2K_GT, keyframe_pose2pose_GT, keyframe_old_K_GT, keyframe_inv_old_K_GT, keyframe_old_pose_GT, keyframe_inv_old_pose_GT;
   
    cv::Matx61f keyframe_pose2pose_algebra, keyframe_pose_algebra;    // NB 'pose' = absolute pose in world coords. 'pose2pose' = from previous keyframe.
    //uint keyframe_num;
    
    
    // image parameters
    cv::Size    base_image_size;
    int         base_image_type;
    
    
    // data files
    std::string rootpath;
    fs::path root;
    vector<fs::path> txt;
    vector<fs::path> png;
    vector<fs::path> depth;
    
    // functions
    void initialize_camera();
    void report_GT_pose_error();
    void display_frame_resluts();
    
    void getPose();     // cv::Mat R, cv::Mat T, cv::Matx44f& pose
    void getInvPose();  // cv::Matx44f pose, cv::Matx44f& inv_pose
    cv::Matx44f getPose(cv::Mat R, cv::Mat T);
    cv::Matx44f getInvPose(cv::Matx44f pose);
    
    int  nextFrame();
    void optimize_depth();
    
    void predictFrame();
    void getFrame();
    void getFrameData();
    void use_GT_pose();
    void artificial_SO3_pose_error();
    void artificial_pose_error();
    
    void generate_invK();
    cv::Matx44f generate_invK_(cv::Matx44f K_);
    void generate_invPose();
    void generate_SE3_k2k( float _SE3_k2k[96] );
    void update_k2k(Matx61f update_);
    void cout_update(int chk_num, int iter, int layer, Matx61f update, float Rho_sq_result);
    void estimateSO3();
    void estimateSE3();
    void estimateCalibration();
    
    void initialize_keyframe_from_GT();
    void initialize_keyframe_from_tracking();
    void initialize_new_keyframe();
    
    void updateDepthCostVol();                 // Built forwards. Updates keframe only when needed.
    void buildDepthCostVol_fast_peripheral();  // Higher levels only, built on current frame.
    //void computeSigmas(float epsilon, float theta, float L, float& sigma_d, float& sigma_q); // use RunCL.cpp version.
    void updateQD();
    void cacheGValues();
    bool updateA();
    
    void SpatialCostFns();
    void ParsimonyCostFns();
    void ExhaustiveSearch();
    
    void getResult();
  
    // return the filenames of all files that have the specified extension
    // in the specified directory and all subdirectories
    void get_all(const fs::path& root , const string& ext, vector<fs::path>& ret ) {  
      if (!fs::exists(root))        return;
      if (fs::is_directory(root))   {
        typedef std::set<boost::filesystem::path> Files;
        Files files;
        fs::recursive_directory_iterator it0(root);
        fs::recursive_directory_iterator endit0;
        std::copy(it0, endit0, std::inserter(files, files.begin()));
        Files::iterator it= files.begin();
        Files::iterator endit= files.end();
        
        while(it != endit)
        {
          if (fs::is_regular_file(*it) && (*it).extension() == ext)
          {
            ret.push_back(*it);
          }
          ++it;
        }
      }
    }
    //void get_all(fs::path& root, const string& ext, vector<fs::path>& ret);
    

  private:
    float old_theta, theta, thetaStart, thetaStep, thetaMin, epsilon, lambda, sigma_d, sigma_q;

};
