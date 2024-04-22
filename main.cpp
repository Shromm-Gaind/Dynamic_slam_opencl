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

using namespace cv;
using namespace std;

////// headers
typedef map<string,int>				int_map;
typedef map<string,float>			float_map;
typedef map<string,vector<float> >	float_vec_map;
typedef map<string,string>			string_map;

class json_params {
	static const int_map 		verbosity;
	static const int_map 		int_;
	static const float_map 		float_;
	static const float_vec_map 	float_vec;
	static const string_map 	paths;		// should I use boost file path ?

public:
	json_params(char * arg);
	void read_verbosity(	Json::Value verbosity_obj);	//,  const int_map 		&verbosity_map);
	void read_paths(		Json::Value paths_obj);		//,  	const string_map 	&path_map
	void read_jparams(		Json::Value params_obj); 	// , 	const int_map		&int_params, 	const float_map		&flt_params, 	const float_vec_map 	&flt_arry_params
	void display_params();
};

/////// main
int main(int argc, char *argv[])
{
	if (argc !=2) { cout << "\n\nUsage : DTAM_OpenCL <config_file.json>\n\n" << flush; exit(1); }
	json_params j_params(argv[1]);

	j_params.display_params();

	exit(0); // TODO just for testing.

}
/*
{
	int verbosity_ 		= obj["verbosity"]["verbosity"].asInt() ;						// -1= none, 0=errors only, 1=basic, 2=lots.
	int imagesPerCV 	= obj["params"]["imagesPerCV"].asUInt() ;
	int max_frame_count = obj["params"]["max_frame_count"].asUInt();
	int frame_count 	= 0;
	int ds_error 		= 0;
																			if(verbosity_>0) cout << "\n\n main_chk 0\n" << flush;
																			cout <<" conf file = " << argv[1] << endl;
																			//cout << ifs.str() <<endl << endl;
																			cout <<"verbosity_ = "<<verbosity_<<", imagesPerCV = "<<imagesPerCV <<endl;
																			cout <<"outpath = " <<  obj["out_path"].asString() <<"\n"<< std::flush;
	
	Dynamic_slam   dynamic_slam(obj);										// Instantiate Dynamic_slam object before while loop.
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
		//dynamic_slam.optimize_depth(); // Temporarily suspend mapping
		//																	if(verbosity_>0) dynamic_slam.runcl.saveCostVols(imagesPerCV);
		
		//dynamic_slam.initialize_keyframe_from_tracking();
		break; // TODO write new depthmap transformation based on bin sort from fluids_v3 & Morphogenesis.
		
	}while(!ds_error && ((frame_count<max_frame_count) || (max_frame_count==-1)) );
																			if(verbosity_>0) cout << "\n main_chk 3\n" << flush;
	dynamic_slam.getResult();												// also calls RunCL::CleanUp()
	exit(0);																// guarantees class destructors are called.
}
*/

/////// functions
json_params::json_params(char * arg){
	//json_params params;
	ifstream ifs(arg);

	Json::Reader reader;
	Json::Value Jval[3];
	map<string, Json::Value> obj{ {"paths",Jval[0]}, {"params",Jval[1]}, {"verbosity",Jval[2]} };

    bool b = reader.parse(ifs, obj["paths"]); 								if (!b) { cout << "Error: " << reader.getFormattedErrorMessages(); exit(1) ;}   else {cout << "NB lists .json file entries alphabetically: \n" << obj["paths"] ;}

	ifstream ifs_params(	obj["paths"]["params_conf"].asString() 	);
	ifstream ifs_verbosity( obj["paths"]["verbosity_conf"].asString()	);

	b = reader.parse(ifs_params, 	obj["params"]); 						if (!b) { cout << "Error: " << reader.getFormattedErrorMessages(); exit(1) ;}   else {cout << "NB lists .json file entries alphabetically: \n" << obj["params"] ;}
	b = reader.parse(ifs_verbosity, obj["verbosity"]); 						if (!b) { cout << "Error: " << reader.getFormattedErrorMessages(); exit(1) ;}   else {cout << "NB lists .json file entries alphabetically: \n" << obj["verbosity"] ;}

	// void read_verbosity(	Json::Value verbosity_obj,  int_map 	&verbosity_map);

	read_verbosity( obj["verbosity"]); 	//params.  , 	verbosity
	read_paths(		obj["paths"]); 		// ,  		paths
	read_jparams(	obj["params"]);		//, int_, float_, float_vec

	//return params;
}


void json_params::read_verbosity(Json::Value verbosity_obj){ // ,  const int_map verbosity_map_
	// iterate over all entries in the verbosity.json file.
	// NB function name clashes between classes ?
	int_map verbosity_map = const_cast<int_map&>(verbosity);

	Json::ArrayIndex size = verbosity_obj.size();
	Json::Value::Members members = verbosity_obj.getMemberNames();
	for (int index=0; index<size; index++){ verbosity_map[  members[index] ]   =    verbosity_obj[index].asInt(); }
}


void json_params::read_paths(Json::Value paths_obj){ //,  const string_map path_map_		// Problem naming fo the and variable as too similar in structure
	string_map path_map = const_cast<string_map&>(paths);

	Json::ArrayIndex size = paths_obj.size();
	Json::Value::Members members = paths_obj.getMemberNames();
	for (int index=0; index<size; index++){ path_map[  members[index] ]   =    paths_obj[index].asString(); }
}


void json_params::read_jparams(Json::Value params_obj){   // , int_map	&int_params, float_map &flt_params, float_vec_map &flt_arry_params  // int_,	float_,	float_vec;
	float_vec_map 	float_vec_mp 	= const_cast<float_vec_map&>(float_vec);
	float_map 		float__mp 		= const_cast<float_map&>(float_);
	int_map 		int__mp 		= const_cast<int_map&>(int_);

	Json::ArrayIndex size = params_obj.size();
	Json::Value::Members members = params_obj.getMemberNames();

	for (int index=0; index<size; index++){

		if (params_obj[index].isArray() ){
			Json::ArrayIndex size2 = params_obj[index].size();

			for (int index2=0; index2<size2; index2++){
				if (params_obj[index][index2].isDouble()) {
					float_vec_mp[  members[index] ][index2]	= params_obj[index][index2].asFloat();			// flt_arry_params
				}
				else {
					cout << "\nparams_obj[\""<< members[index] <<"\"] : "<<  params_obj[index].asString()  << " , is not a float_array"<<flush;
					break;
				}
			}
		}
		else if (params_obj[index].isDouble()){
			float__mp[  members[index] ]	= params_obj[index].asFloat();  		// flt_params	float_
		}
		else if (params_obj[index].isInt() ){
			int__mp[  members[index] ]   	= params_obj[index].asInt(); 			// int_params	int_
		}
		else {
			cout << "\nparams_obj[\""<< members[index] <<"\"] : "<<  params_obj[index].asString()  << " , is not a float_array, float or int."<<flush;
		}
	}
}


void json_params::display_params(  ) {
	cout << "\n\n params.paths   = "  << flush;
	for (auto elem : paths) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;

	cout << "\n\n params.int_   = "  << flush;
	for (auto elem : int_) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;

	cout << "\n\n params.float_   = "  << flush;
	for (auto elem : float_) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;

	cout << "\n\n params.float_vec   = "  << flush;
	for (auto elem : float_vec) {
		cout << "\n " <<  elem.first <<" :  {";
		for (auto elem2 : elem.second) {
			cout << elem2 << ", ";
		}
		cout << flush;
	}

	cout << "\n\n params.verbosity   = "  << flush;
	for (auto elem : verbosity) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;
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
