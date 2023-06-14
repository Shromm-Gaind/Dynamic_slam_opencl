#include "Dynamic_slam.h"
#include "utils/fileLoader.hpp"
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;
int main(int argc, char *argv[])
{
	if (argc !=2) { cout << "\n\nUsage : DTAM_OpenCL <config_file.json>\n\n" << flush; exit; }
	ifstream ifs(argv[1]);
    Json::Reader reader;
    Json::Value obj;
    bool b = reader.parse(ifs, obj);
	if (!b) { cout << "Error: " << reader.getFormattedErrorMessages(); }   else {cout << "NB lists .json file entries alphabetically: \n" << obj ;}
	int verbosity_ 		= obj["verbosity"].asInt() ;						// -1= none, 0=errors only, 1=basic, 2=lots.
	int imagesPerCV 	= obj["imagesPerCV"].asUInt() ;
																			if(verbosity_>0) cout << "\n\n main_chk 0\n" << flush;
																			cout <<" conf file = " << argv[1] << endl;
																			//cout << ifs.str() <<endl << endl;
																			cout <<"verbosity_ = "<<verbosity_<<", imagesPerCV = "<<imagesPerCV <<endl;
																			cout <<"outpath  " <<  obj["out_path"].asString() << std::flush;
	
	Dynamic_slam dynamic_slam(obj);											// Instantiate Dynamic_slam object before while loop.
																			if(verbosity_>0) cout << "\n main_chk 1\n" << flush;
	// New continuous while loop: load next (image + data), Dynamic_slam::nextFrame(..)
	int max_frame_count = obj["max_frame_count"].asUInt();	
	int frame_count 	= 0;
	int ds_error 		= 0;
	do{
		ds_error = dynamic_slam.nextFrame();
		frame_count ++;
	}while(!ds_error && ((frame_count<max_frame_count) || (max_frame_count==-1)) );
																			if(verbosity_>0) cout << "\n main_chk 2\n" << flush;
	dynamic_slam.getResult();												// also calls RunCL::CleanUp()
	exit(0);																// guarantees class destructors are called.
}
	
	
	
	
/*	
{	
	///////////////////////////////////////////////////////
	
	for (int i =0; i < imagesPerCV; i++) {									// Load images & data from file into c++ vectors
		loadAhanda( ss0.str(),
					(i*incr)+offset,
					image,													// NB .png image is converted CV_8UC3 -> CV_32FC3, and given GaussianBlurr (3,3).
					d,
					cameraMatrix,											// NB recomputes cameraMatrix for each image, but hard codes 640x480.
					R,
					T);
																			if(verbosity_>0) cout << ", image.size="<<image.size << "\ncameraMatrix.rows="<<cameraMatrix.rows<<",  cameraMatrix.cols="<<cameraMatrix.rows<<"\n"<<flush;
		images.push_back(image);
		Rs.push_back(R.clone());
		Ts.push_back(T.clone());
		ds.push_back(d.clone());
		Rs0.push_back(R.clone());
		Ts0.push_back(T.clone());
		D0.push_back(1 / d);
	}
																			if(verbosity_>0){
																				cout<<"\ncameraMatrix.type()="<<cameraMatrix.type()<< "\ntype 5 = CV_32FC1\n"<<std::flush;
																				int rows = cameraMatrix.rows;
																				int cols = cameraMatrix.cols;
																				
																				for (int i=0; i<rows; i++){
																					for (int j=0; j<cols; j++){
																						cout<<", "<<cameraMatrix.at<float>(i,j)<< std::flush;
																					}cout<<"\n"<< std::flush;
																				}cout<<"\n"<< std::flush;
																				cout << "\n main_chk 3\n" << flush; 							//Setup camera matrix
																			}
	int   startAt 				= obj["startAt"].asUInt();					if(verbosity_>0) cout<<"images[startAt].size="<<images[startAt].size<<"\n";
																			if(verbosity_>0) cout << "\n main_chk 3.1\n" << flush; 				// Instantiate CostVol ///
	
	
	CostVol cv(images[startAt], Rs[startAt], Ts[startAt], cameraMatrix, obj);
																			if(verbosity_>0) cout << "\n main_chk 4\tcalculate cost volume: ================================================" << endl << flush;
	
																			
	for (int imageNum = 1; imageNum < imagesPerCV; imageNum+=1){			// Update CostVol ////////////////
		cv.updateCost(images[imageNum], Rs[imageNum], Ts[imageNum]);
																			if(verbosity_>0) {
																				cout<<"\ncv.updateCost: images["<<imageNum<<"].size="<<images[imageNum].size<<"\n";
																				if (imageNum%5 == 1) cv.cvrc.saveCostVols(imageNum+1);
																			}
	}
	
	
																			if(verbosity_>0) cout << "\n main_chk 5\tcacheGValues: =========================================================" << endl<<flush;
	cv.cacheGValues();														// cacheGValues()  elementwise weighting on keframe image gradient
																			if(verbosity_>0) cout << "\n main_chk 6\toptimizing: ===========================================================" << endl<<flush;
	
	
	bool doneOptimizing;
	int  opt_count 		= 0;
	int  max_opt_count 	= obj["max_opt_count"].asInt();
	do{ 
		for (int i = 0; i < 10; i++) cv.updateQD();							// Optimize Q, D   (primal-dual)
		doneOptimizing = cv.updateA();										// Optimize A      (pointwise exhaustive search)
		opt_count ++;
	} while (!doneOptimizing && (opt_count<max_opt_count));
																			if(verbosity_>0) cout << "\n main_chk 7\n" << flush;
	
	
	
	
	cv::Mat depthMap;
	cv.GetResult();															// GetResult /////////////////////
    depthMap 		= cv._a;
	double minVal	=1, 	maxVal=1;
	cv::Point minLoc={0,0}, maxLoc{0,0};
	cv::minMaxLoc(depthMap, &minVal, &maxVal, &minLoc, &maxLoc);
	std::stringstream ss;
	ss << "Depthmap, maxVal: " << maxVal << " minVal: " << minVal << " minLoc: "<< minLoc <<" maxLoc: " << maxLoc;
	cv::imshow(ss.str(), (depthMap*(1.0/maxVal)));
																			if(verbosity_>0) std::cout << ss.str() << std::endl<<flush;
	cv::waitKey(-1);														cout << "\n main_chk 9 : finished\n" << flush;
	exit(0);
}
*/
