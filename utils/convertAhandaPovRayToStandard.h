#ifndef CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
#define CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED

#include <opencv2/opencv.hpp>
#include "conf_params.hpp"

void convertAhandaPovRayToStandard(int_map verbosity_mp, const char * filepath, cv::Mat& R,  cv::Mat& T,  cv::Mat& cameraMatrix);

cv::Mat loadDepthAhanda(int_map verbosity_mp, std::string filename, int r,int c,cv::Mat cameraMatrix);

#endif // CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
