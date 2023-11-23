#ifndef CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
#define CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED

#include <opencv2/opencv.hpp>
void convertAhandaPovRayToStandard(const char * filepath, cv::Mat& R,  cv::Mat& T,  cv::Mat& cameraMatrix);

cv::Mat loadDepthAhanda(std::string filename, int r,int c,cv::Mat cameraMatrix);

#endif // CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
