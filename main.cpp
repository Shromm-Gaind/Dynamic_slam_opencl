#include "Dynamic_slam/Dynamic_slam.h"
//#include "utils/fileLoader.hpp"
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>

#include <map>
#include <string>
#include <string_view>

#include <string>
#include <sstream>
#include "Dynamic_slam/Dynamic_slam.h"
#include "utils/conf_params.hpp"

using namespace cv;
using namespace std;


void copy_conf(Json::String source_filepath,  Json::String infile,  string outfile   ){
	filesystem::path in_path_verbosity(  source_filepath  +  infile  );
	filesystem::path out_path_verbosity( outfile   );
	out_path_verbosity.replace_filename( in_path_verbosity.filename() );
	cerr << "\nin_path="<<in_path_verbosity<<",  out_path="<<out_path_verbosity<<endl<<flush;
	filesystem::copy( in_path_verbosity , out_path_verbosity );
}

/////// main
int main(int argc, char *argv[])
{
	if (argc !=2) { cout << "\n\nUsage : DTAM_OpenCL <config_file.json>\n\n" << flush; exit(1); }

	Json::Value obj;
	conf_params j_params(argv[1], obj);
	j_params.display_params();

	int verbosity_ 		= j_params.verbosity_mp["verbosity"]; 				//obj["verbosity"]["verbosity"].asInt() ;			//  					// -1= none, 0=errors only, 1=basic, 2=lots.
	int imagesPerCV 	= obj["imagesPerCV"].asUInt() ;						// j_params.int_mp["imagesPerCV"]; 			//
	int max_frame_count = obj["max_frame_count"].asUInt();					// j_params.int_mp["max_frame_count"]; 		//
	int frame_count 	= 0;
	int ds_error 		= 0;
																			if(verbosity_>0) cout << "\n\n main_chk 0\n" << flush;
																			cout <<"\nconf file = "		<< argv[1];
																			cout <<"\nverbosity_ = "	<<verbosity_;
																			cout <<"\nimagesPerCV = "	<<imagesPerCV;
																			cout <<"\noutpath = " 		<< j_params.paths_mp["out_path"];

	Dynamic_slam   dynamic_slam(obj, j_params.verbosity_mp);				// Instantiate Dynamic_slam object before while loop.

	stringstream outfile;													// Redirecting cout to write to "output.txt"
	outfile << dynamic_slam.runcl.paths.at("folder").c_str() <<  "Dynamic_slam_output.txt";
    cout << "\nOutfile = " << outfile.str() << endl;
    fflush (stdout);
    freopen (outfile.str().c_str(), "w", stdout);
	cout << "Dynamic_slam started. Objects created. Initializing.\n\n" << flush;

	// save config files to output
	//bool copy_file(const path& from, const path& to, copy_options options);   				 // Boost
	//void copy( 	const std::filesystem::path& from	,  const std::filesystem::path& to 	);   // c++17 <filesystem>

	// filesystem::path in_path_params(  obj["source_filepath"].asString()  +  obj["params_conf"].asString()  );
	// filesystem::path out_path_params( outfile.str().c_str() );
	// out_path_params.replace_filename( in_path_params.filename() );
	// cerr << "\nin_path="<<in_path_params<<",  out_path="<<out_path_params<<endl<<flush;
	// filesystem::copy( in_path_params , out_path_params );

	copy_conf( obj["source_filepath"].asString(),  obj["params_conf"].asString(),		outfile.str().c_str()  );
	copy_conf( obj["source_filepath"].asString(),  obj["verbosity_conf"].asString(),	outfile.str().c_str()  );
	copy_conf( ""								,  argv[1],    							outfile.str().c_str()  );


	//boost::filesystem::copy_file( obj["verbosity_conf"]  ,  outfile.str().c_str()      );
																			if(verbosity_>0) cout << "\n main_chk 1\n" << flush;
																			// New continuous while loop: load next (image + data), Dynamic_slam::nextFrame(..)
																			// NB need to initialize the cost volume with a key frame, and tracking with a depth map.
																			// Also need a test for when to start a new keyframe.
	dynamic_slam.initialize_keyframe_from_GT();
	frame_count++;
																			if(verbosity_>0) cout << "\n main_chk 2\n" << flush;
	do{																		// Long do while not yet crashed loop.
		for (int i=0; i<imagesPerCV ; i++){									// Inner loop per keyframe.params
			ds_error = dynamic_slam.nextFrame();
			frame_count ++;
		}
		dynamic_slam.optimize_depth(); // Temporarily suspend mapping
		//																	if(verbosity_>0) dynamic_slam.runcl.saveCostVols(imagesPerCV);
		
		//dynamic_slam.initialize_keyframe_from_tracking();
		break; // TODO write new depthmap transformation based on bin sort from fluids_v3 & Morphogenesis.
		
	}while(!ds_error && ((frame_count<max_frame_count) || (max_frame_count==-1)) );
																			if(verbosity_>0) cout << "\n main_chk 3\n" << flush;
	dynamic_slam.getResult();												// also calls RunCL::CleanUp()

	cout << "\n\nDynamic_slam finished. Exiting."<<flush;
	fflush (stdout);
    fclose (stdout);
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
