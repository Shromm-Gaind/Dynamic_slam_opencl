#pragma once

#include <string>
#include <boost/filesystem.hpp>
#include <fstream>
#include <set>

//#include "utils/utils.hpp"
#include "RunCL.h"

namespace fs = ::boost::filesystem;

class Dynamic_slam
{
public:
    ~Dynamic_slam();
    Dynamic_slam(
        Json::Value obj_
    );
    
    Json::Value obj;
    RunCL runcl;
    int verbosity;
    int frame_num;
    cv::Mat image, R, T, d, cameraMatrix;
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
    int  nextFrame();
    void predictFrame();
    void getFrame();
    void getFrameData();
    void estimateSO3();
    void estimateSE3();
    void estimateCalibration();
    void buildDepthCostVol();
    void SpatialCostFns();
    void ParsimonyCostFns();
    void ExhaustiveSearch();
    
    void getResult();
    
    
    #define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root , const string& ext, vector<fs::path>& ret )
{  
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
