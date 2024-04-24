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
/*
////// headers
typedef map<string,bool>						bool_map;
typedef map<string,int>							int_map;
typedef map<string,float>						float_map;
typedef map<string,vector<float> >				float_vec_map;
typedef map<string,vector<vector<float>> >		float_vecvec_map;
typedef map<string,string>						string_map;
typedef map<string,vector<string>> 				string_vec_map;

class conf_params {
	int_map 					verbosity_mp;
	bool_map					bool_mp;
	int_map 					int_mp;
	float_map 					float_mp;
	float_vec_map 				float_vec_mp;
	float_vecvec_map 			float_vecvec_mp;
	string_vec_map				string_vec_mp;
	string_map 					paths_mp;				// should I use boost file path ?

public:
	conf_params(char * arg);
	void read_verbosity(	Json::Value verbosity_obj);	//,  const int_map 		&verbosity_map);
	void read_paths(		Json::Value paths_obj);		//,  const string_map 	&path_map
	void read_jparams(		Json::Value params_obj); 	//,  const int_map		&int_params, 	const float_map		&flt_params, 	const float_vec_map 	&flt_arry_params

	void readVecVecFloat(  	string member, Json::Value params_obj );
	void readVecFloat( 		string member, Json::Value params_obj );
	void readVecString( 	string member, Json::Value params_obj );

	void display_params();
};
*/
/////// main
int main(int argc, char *argv[])
{
	if (argc !=2) { cout << "\n\nUsage : DTAM_OpenCL <config_file.json>\n\n" << flush; exit(1); }
	Json::Value obj;
	conf_params j_params(argv[1], obj);
	j_params.display_params();
	//cout << "\n obj = " << obj << endl << flush;

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


/*
/////// functions
conf_params::conf_params(char * arg){
	//json_params params;
	ifstream ifs(arg);

	Json::Reader reader;
	Json::Value paths_obj, params_obj, verbosity_obj;
	//map<string, Json::Value> obj{ {"paths",Jval[0]}, {"params",Jval[1]}, {"verbosity",Jval[2]} };

    bool b = reader.parse(ifs, paths_obj); 										if (!b) { cout << "Error: " << reader.getFormattedErrorMessages(); exit(1) ;}   else {cout << "NB lists .json file entries alphabetically: \n" << paths_obj ;}
																				cout << "\njson_params::json_params(char * arg) chk 1"<<flush;



	ifstream ifs_params(	paths_obj["source_filepath"].asString()	 +  paths_obj["params_conf"].asString() 	);	if (!ifs_params.is_open()) {
																															cout << "\njson_params::json_params(char * arg):  ifs_params  is NOT open !"<< flush;
																															cout << "\nfilepath + "<< paths_obj["source_filepath"].asString()	 +  paths_obj["params_conf"].asString() << endl<<flush;
																															exit(1);
																													}

	ifstream ifs_verbosity( paths_obj["source_filepath"].asString()	 +  paths_obj["verbosity_conf"].asString()	);	if (!ifs_verbosity.is_open()) {
																															cout << "\njson_params::json_params(char * arg):  ifs_verbosity  is NOT open !"<< flush;
																															cout << "\nfilepath + "<< paths_obj["source_filepath"].asString()	 +  paths_obj["verbosity_conf"].asString() << endl<<flush;
																															exit(1);
																													}

	b = reader.parse(ifs_params, 	params_obj); 								if (!b) { cout << "Error: " << reader.getFormattedErrorMessages(); exit(1) ;}   else {cout << "NB lists .json file entries alphabetically: \n" << params_obj ;}

																				cout << "\njson_params::json_params(char * arg) chk 4\n"<<flush;

	b = reader.parse(ifs_verbosity, verbosity_obj); 							if (!b) { cout << "Error: " << reader.getFormattedErrorMessages(); exit(1) ;}   else {cout << "NB lists .json file entries alphabetically: \n" << verbosity_obj ;}

																				cout << "\njson_params::json_params(char * arg) chk 5\n"<<flush;

																				// void read_verbosity(	Json::Value verbosity_obj,  int_map 	&verbosity_map);

	read_verbosity( verbosity_obj); 											//params.  , 	verbosity
																				cout << "\njson_params::json_params(char * arg) chk 6\n"<<flush;

	read_paths(		paths_obj); 												// ,  		paths
																				cout << "\njson_params::json_params(char * arg) chk 7\n"<<flush;

	read_jparams(	params_obj);												//, int_, float_, float_vec
																				cout << "\njson_params::json_params(char * arg) chk 8\n"<<flush;

																				//return params;
}


void conf_params::read_verbosity(Json::Value verbosity_obj){ // ,  const int_map verbosity_map_
	// iterate over all entries in the verbosity.json file.
	// NB function name clashes between classes ?
																													cout << "\njson_params::read_verbosity(..) : chk_0\n"<<flush;
	Json::Value::ArrayIndex size = verbosity_obj.size();
	Json::Value::Members members = verbosity_obj.getMemberNames();
																													cout << "\njson_params::read_verbosity(..) : chk_2\n"<<flush;
																													cout << "\nsize = "<< size << flush;
																													cout << "\nmembers[0] = "<< members[0] <<flush;
	string member;
	for (int index=0; index<size; index++){
		member = members[index];
		verbosity_mp[  members[index] ]   =    verbosity_obj[member].asInt();
	}
																													cout << "\njson_params::read_verbosity(..) : finished\n"<<flush;
}


void conf_params::read_paths(Json::Value paths_obj){ //,  const string_map path_map_		// Problem naming fo the and variable as too similar in structure
	//string_map path_map = const_cast<string_map&>(paths);

	Json::ArrayIndex size = paths_obj.size();
	Json::Value::Members members = paths_obj.getMemberNames();
	string member;
	for (int index=0; index<size; index++){
		member = members[index];
		paths_mp[  members[index] ]   =    paths_obj[member].asString();

	}
}


void conf_params::read_jparams(Json::Value params_obj){   // , int_map	&int_params, float_map &flt_params, float_vec_map &flt_arry_params  // int_,	float_,	float_vec;
																													cout << "\njson_params::read_jparams(..) : chk_0\n"<<flush;
	Json::ArrayIndex size = params_obj.size();
	Json::Value::Members members = params_obj.getMemberNames();
	string member;
																													//cout << "\njson_params::read_jparams(..) : chk_1\n"<<flush;
	for (int index=0; index<size; index++){
		member = members[index];
																													//cout << "\nindex="<<index<<flush; cout<< "\tmember="<<member<<" ,"<<flush;
		if (params_obj[member].isArray() ){ 																		//cout << "\n isArray() " << members[index] <<" : "<< flush;
			Json::ArrayIndex size2 = params_obj[member].size();														//cout << "\n array size = "<< size2 << " , " << flush;
			if (params_obj[member][0].isString() ){																	//cout << "\n isArray() " << members[index] <<" : "<< flush;
				readVecString( member, params_obj  );
			}
			else if (params_obj[member][0].isArray() ){																//cout << "\n isArray() " << members[index] <<" : "<< flush;
				if (params_obj[member][0][0].isNumeric() ) {
					readVecVecFloat(  member, params_obj );
				}
				else cout << "\n " << member << " is not float_vecvec  :    "  << flush;
			}
			else if (params_obj[member][0].isNumeric() ) {															//cout << "\nisNumeric"<<flush;
				readVecFloat( member, params_obj );
			}
			else {
				cout << "\nparams_obj[\""<< members[index] <<"\"] : "<<  params_obj[member].asString()  << " , is not a float_array. break;"<<flush;
			}
		}
		else if (params_obj[member].isBool()){																		//cout << "\n isBool() " << members[index] <<" : "<< params_obj[member].asBool() <<" , "<< flush;
			bool_mp[  members[index] ]	= params_obj[member].asBool();
		}
		else if (params_obj[member].isInt() ){																		//cout << "\n isInt() " << members[index] <<" : "<< params_obj[member].asInt() <<" , "<< flush;
			int_mp[  members[index] ]   	= params_obj[member].asInt();
		}
		else if (params_obj[member].isDouble() && ! params_obj[member].isInt() ){  									//cout << "\n isDouble() " << members[index] <<" : "<< params_obj[member].asDouble() <<" , "<< flush;
			float_mp[  members[index] ]	= params_obj[member].asFloat();
		}
		else {
			cout << "\nparams_obj[\""<< members[index] <<"\"] : "<<  params_obj[member].asString()  << " , is not a float_array, float or int."<<flush;
		}
	}
																													cout << "\njson_params::read_jparams(..) : finished\n"<<flush;
}


void conf_params::readVecVecFloat( string member, Json::Value params_obj  ){
	vector<vector<float>> vec0;
	Json::ArrayIndex size2 = params_obj[member].size();																cout << "\n array size = "<< size2 << " , " << flush;
	for (int index2=0; index2<size2; index2++){																		cout << "\nindex2 = "<< index2 ;
		vector<float> vec1;
		Json::ArrayIndex size3 = params_obj[member].size();															cout << "\n array size = "<< size3 << " , " << flush;

		for (int index3=0; index3<size2; index3++){																	cout << "\nindex3 = "<< index3 ;
			if (params_obj[member][index2][index3].isNumeric() ) {													cout << "\nisNumeric"<<flush;
																													cout << "\nparams_obj[member][index2][index3] = " << params_obj[member][index2][index3].asFloat() << " , " << flush;
			vec1.push_back( params_obj[member][index2][index3].asFloat() );
			}
		}
		vec0.push_back(vec1);
	}
	float_vecvec_mp[  member ] = vec0;
}

void conf_params::readVecFloat( string member, Json::Value params_obj  ){
	vector<float> vec0;
	Json::ArrayIndex size2 = params_obj[member].size();																cout << "\n array size = "<< size2 << " , " << flush;
	for (int index2=0; index2<size2; index2++){																		cout << "\nindex2 = "<< index2 ;
		if (params_obj[member][index2].isNumeric() ) {																cout << "\nisNumeric"<<flush;
																													cout << "\nparams_obj[member][index2][index3] = " << params_obj[member][index2].asFloat() << " , " << flush;
			vec0.push_back( params_obj[member][index2].asFloat() );
		}
	}
	float_vec_mp[  member ] = vec0;
}


void conf_params::readVecString( string member, Json::Value params_obj  ){
	vector<string> vec0;
	Json::ArrayIndex size2 = params_obj[member].size();																cout << "\n array size = "<< size2 << " , " << flush;
	for (int index2=0; index2<size2; index2++){																		cout << "\nindex2 = "<< index2 ;
																													cout << "\nparams_obj[member][index2][index3] = " << params_obj[member][index2].asString() << " , " << flush;
			vec0.push_back( params_obj[member][index2].asString() );
	}
	string_vec_mp[  member ] = vec0;
}


void conf_params::display_params(  ) {
	cout << "\n\n params.paths   = "  << flush;
	for (auto elem : paths_mp) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;

	cout << "\n\n params.int_   = "  << flush;
	for (auto elem : int_mp) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;

	cout << "\n\n params.float_   = "  << flush;
	for (auto elem : float_mp) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;

	cout << "\n\n params.float_vec   = "  << flush;
	for (auto elem : float_vec_mp) {
		cout << "\n " <<  elem.first <<" :  {";
		for (auto elem2 : elem.second) {
			cout << elem2 << ", ";
		}
		cout << " } " << flush;
	}

	cout << "\n\n params.float_vecvec   = "  << flush;
	for (auto elem : float_vecvec_mp) {
		cout << "\n " <<  elem.first <<" :  {";
		for (auto elem2 : elem.second) {
			cout <<"\t [ " ;
			for (auto elem3 : elem2){
				cout << elem3 << ", ";
			}
			cout <<"], ";
		}
		cout << " } " << flush;
	}

	cout << "\n\n params.string_vec   = "  << flush;
	for (auto elem : string_vec_mp) {
		cout << "\n " <<  elem.first <<" :  {";
		for (auto elem2 : elem.second) {
			cout << elem2 << ", ";
		}
		cout << " } " << flush;
	}

	cout << "\n\n params.verbosity   = "  << flush;
	for (auto elem : verbosity_mp) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;
}
*/

	
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
