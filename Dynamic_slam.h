#pragma once

#include <string>
#include <boost/filesystem.hpp>
#include <fstream>
#include <set>
#include "convertAhandaPovRayToStandard.h"
#include "RunCL.h"

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
    
    // camera & pose params
    cv::Matx44f K, inv_K, pose, inv_pose, K2K, pose2pose, old_K, inv_old_K, old_pose, inv_old_pose, transform[6]  ;
    cv::Matx61f pose2pose_algebra_0, pose2pose_algebra_1, pose2pose_algebra_2;
    
    cv::Matx44f K_GT, inv_K_GT, pose_GT, inv_pose_GT, K2K_GT, pose2pose_GT, old_K_GT, inv_old_K_GT, old_pose_GT, inv_old_pose_GT, transform_GT[6]  ; // Ground Truth from the dataset.
    
    cv::Mat image, R, T, depth_GT, cameraMatrix, projection;       // TODO should these be Matx ? 
    cv::Mat old_R, old_T, R_dif, T_dif;
    float SE3_k2k[6*16];
    
    // lens distortion params
    
    
    // 
    
    
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
    void getPose(); // cv::Mat R, cv::Mat T, cv::Matx44f& pose
    void getInvPose(); // cv::Matx44f pose, cv::Matx44f& inv_pose
    cv::Matx44f getPose(cv::Mat R, cv::Mat T);
    cv::Matx44f getInvPose(cv::Matx44f pose);
    
    int  nextFrame();
    void predictFrame();
    void getFrame();
    void getFrameData();
    void use_GT_pose();
    void artificial_pose_error();
    
    void generate_invK();
    void generate_invPose();
    void generate_SE3_k2k( float _SE3_k2k[96] );
    void estimateSO3();
    void estimateSE3();
    void estimateCalibration();
    void buildDepthCostVol();
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

};
