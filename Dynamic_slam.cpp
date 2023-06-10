#include "Dynamic_slam.h"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;


Dynamic_slam::~Dynamic_slam()
{
};

Dynamic_slam::Dynamic_slam( Json::Value obj_ ): runcl(obj_) {
	int local_verbosity_threshold = 1;
	obj = obj_;
	verbosity 			= obj["verbosity"].asInt();
	runcl.frame_num 	= obj["data_file_offset"].asUInt();
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 0\n" << flush;
	stringstream ss0;
	ss0 << obj["data_path"].asString() << obj["data_file"].asString();
	rootpath = ss0.str();
	root = rootpath;
	
	if ( exists(root)==false )		{ cout << "Data folder "<< ss0.str()  <<" does not exist.\n" <<flush; exit(0); }
	if ( is_directory(root)==false ){ cout << "Data folder "<< ss0.str()  <<" is not a folder.\n"<<flush; exit(0); }
	if ( empty(root)==true )		{ cout << "Data folder "<< ss0.str()  <<" is empty.\n"		 <<flush; exit(0); }
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 1\n" << flush;
	get_all(root, ".txt",   txt);																						// Get lists of files. Gathers all filepaths with each suffix, into c++ vectors.
	get_all(root, ".png",   png);
	get_all(root, ".depth", depth);
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 2\n" << flush;
																														if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam(): "<< png.size()  <<" .png images found in data folder.\t"<<flush;
	runcl.baseImage 	= imread(png[runcl.frame_num].string());														// Set image params, ref for dimensions and data type.
																														if (verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam_chk 3: runcl.baseImage.size() = "<< runcl.baseImage.size() \
																															<<" runcl.baseImage.type() = " << runcl.baseImage.type() << "\t"<< runcl.checkCVtype(runcl.baseImage.type()) <<flush;
																														if(verbosity>1) { imshow("runcl.baseImage",runcl.baseImage); cv::waitKey(-1); }
	runcl.initialize();
	runcl.allocatemem();
																														if (verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam_chk 4: runcl.baseImage.size() = "<< runcl.baseImage.size() \
																															<<" runcl.baseImage.type() = " << runcl.baseImage.type() << "\t"<< runcl.checkCVtype(runcl.baseImage.type()) <<flush;
	initialize_camera();
	generate_SE3_k2k( SE3_k2k );
	runcl.precom_param_maps( SE3_k2k );
};

void Dynamic_slam::initialize_camera(){
	int local_verbosity_threshold = 0;
	K = K.zeros();																										// NB In DTAM_opencl, "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	K.operator()(3,3) = 1;
	for (int i=0; i<9; i++){K.operator()(i/3,i%3) = obj["cameraMatrix"][i].asFloat(); }
																														if(verbosity>local_verbosity_threshold) {
																															/*
																															cv::Matx44f test_pose = pose * inv_pose;
																															cout<<"\n\ninv_pose\n";									// Verify inv_pose:////////////////////////////////////////////////////
																															for(int i=0; i<4; i++){
																																for(int j=0; j<4; j++){
																																	cout<<"\t"<< std::setw(5)<<inv_pose.operator()(i,j);
																																}cout<<"\n";
																															}cout<<"\n";

																															cout<<"\n\ntest_pose inversion: pose * inv_pose;\n";
																																for(int i=0; i<4; i++){
																																	for(int j=0; j<4; j++){
																																		cout<<"\t"<< std::setw(5)<<test_pose.operator()(i,j);
																																	}cout<<"\n";
																																}cout<<"\n";

																															cout<<"\n\ntest_pose inversion: inv_pose * pose;\n";
																															test_pose = inv_pose * pose;
																															for(int i=0; i<4; i++){
																																for(int j=0; j<4; j++){
																																	cout<<"\t"<< std::setw(5)<<test_pose.operator()(i,j);
																																}cout<<"\n";
																															}cout<<"\n";
																															*/

																															for (int i=0; i<9; i++) {
																																cout<<"\ni="<<i<<","<<flush;
																																cout<<"\tK.operator()(i/3,i%3)="<<K.operator()(i/3,i%3)<<","<<flush;
																																cout<<"\tobj[\"cameraMatrix\"][i].asFloat()="<< obj["cameraMatrix"][i].asFloat() <<","<<flush;
																															}
																														}
	generate_invK();
	
	R 		= cv::Mat::zeros(3,3 , CV_32FC1);																			// intialize ground truth extrinsic data, NB Mat (int rows, int cols, int type)
	T 		= cv::Mat::zeros(3,1 , CV_32FC1);
	R_dif 	= cv::Mat::zeros(3,3 , CV_32FC1);
	T_dif 	= cv::Mat::zeros(3,1 , CV_32FC1);
																														if (verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam_chk 5:" <<flush;
	for (int i=0; i<9; i++){cout << "\n R.at<float>("<<i<<")="<< R.at<float>(i)<< flush ;  }
																														if (verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam_chk 6:" <<flush;
	
	// TODO Also initialize any lens distorsion, vinetting. etc
}

int Dynamic_slam::nextFrame() {
	int local_verbosity_threshold = 1;
	runcl.frame_bool_idx = !runcl.frame_bool_idx;																		// Global array index swap for: cl_mem imgmem[2],  gxmem[2], gymem[2], g1mem[2],  k_map_mem[2], SE3_map_mem[2], dist_map_mem[2];
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::nextFrame_chk 0,  runcl.frame_bool_idx="<<runcl.frame_bool_idx<<"\n" << flush;
	predictFrame();
	getFrame();
	
	estimateSO3();
	estimateSE3(); 				// own thread ? num iter ?
	estimateCalibration(); 		// own thread, one iter.
	
	getFrameData();																										// Loads GT depth of the new frame. NB depends on image.size from getFrame().
	
	buildDepthCostVol();
	
	int outer_iter = obj["regularizer_outer_iter"].asInt();
	int inner_iter = obj["regularizer_inner_iter"].asInt();
	for (int i=0; i<outer_iter ; i++){
		for (int j=0; j<inner_iter ; j++){
			SpatialCostFns();
			ParsimonyCostFns();
		}
		ExhaustiveSearch();
	}
	runcl.frame_num++;
	return(0);					// NB option to return an error that stops the main loop.
};

void Dynamic_slam::predictFrame()
{
// generate initial prediction while data loads. OR do this at the end of the loop ?
	// new_pose = old pose + vel*timestep + accel*timestep²
	
	// kernel update DepthMap with RelVelMap
	
	// kernel predict new frame
};

void Dynamic_slam::getFrame() { // can load use separate CPU thread(s) ?  // NB also need to change type CV_8UC3 -> CV_16FC3
	int local_verbosity_threshold = 1;
																														if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrame_chk 0\n" << flush;
																															// # load next image to buffer NB load at position [log_2 index]
																															// See CostVol::updateCost(..) & RunCL::calcCostVol(..) 
																															cout << "\n\nDynamic_slam::getFrame()";
																															cout << "\nruncl.baseImage.size() =" << runcl.baseImage.size();
																															cout << "\nruncl.baseImage_size =" << runcl.baseImage_size;
																															cout << "\nruncl.baseImage_type =" << runcl.baseImage_type ;
																															cout << "\nruncl.image_size_bytes =" << runcl.image_size_bytes ;
																															cout << endl;
																															cout << "\nruncl.mm_Image_type =" << runcl.mm_Image_type ;
																															cout << "\nruncl.mm_size_bytes_C3 =" << runcl.mm_size_bytes_C3 ;
																															cout << "\nruncl.mm_size_bytes_C1 =" << runcl.mm_size_bytes_C1 ;
																															cout << "\nruncl.mm_vol_size_bytes =" << runcl.mm_vol_size_bytes ;
																															cout << "\nruncl.mm_Image_size =" << runcl.mm_Image_size ;
																															cout << "\n" << flush ;
																														}
	image = imread(png[runcl.frame_num].string());
																														if (image.type()!= runcl.baseImage.type() || image.size()!=runcl.baseImage.size() ) {
																															cout<< "\n\nError: Dynamic_slam::getFrame(), runcl.frame_num = " << runcl.frame_num << " : missmatched. runcl.baseImage.size()="<<runcl.baseImage.size()<<\
																															", image.size()="<<image.size()<<", runcl.baseImage.type()="<<runcl.baseImage.type()<<", image.type()="<<image.type()<<"\n\n"<<flush;
																															exit(0);
																														}
																														//image.convertTo(image, CV_16FC3, 1.0/256, 0.0); // NB cv_16FC3 is preferable, for faster half precision processing on AMD, Intel & ARM GPUs. 
	runcl.loadFrame( image );																							// NB Nvidia GeForce have 'Tensor Compute" FP16, accessible by PTX. AMD have RDNA and CDNA. These need PTX/assembly code and may use BF16 instead of FP16.
																														// load a basic image in CV_8UC3, then convert on GPU to 'half'
	runcl.cvt_color_space( );
	runcl.img_variance();
	runcl.mipmap_linear();																								// (uint num_reductions, uint gaussian_size)// TODO set these as params in conf.json
	runcl.img_gradients();
																														// # Get 1st & 2nd order image gradients of MipMap
																														// see CostVol::cacheGValues(), RunCL::cacheGValue2 & __kernel void CacheG3
																														if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrame_chk 1  Finished\n" << flush;}
}

cv::Matx44f Dynamic_slam::getPose(Mat R, Mat T){	// Mat R, Mat T, Matx44f& pose  // NB Matx::operator()() does not copy, but creates a submatrix. => would be updated when R & T are updated.
	cv::Matx44f pose;
	for (int i=0; i<9; i++) {pose.operator()(i/3,i%3) = 1 * R.at<float>(i/3,i%3); 										cout << "\nR.at<float>("<<i/3<<","<<i%3<<") = " << R.at<float>(i/3,i%3) << ",   pose.operator()(i/3,i%3) = " << pose.operator()(i/3,i%3) ;      }
	for (int i=0; i<3; i++) pose.operator()(i,3)      = T.at<float>(i);
	for (int i=0; i<3; i++) pose.operator()(3,i)      = 0.0f;
	pose.operator()(3,3) = 1.0f;
	return pose;
}

cv::Matx44f Dynamic_slam::getInvPose(cv::Matx44f pose) {	// Matx44f pose, Matx44f& inv_pose
	cv::Matx44f local_inv_pose;
	for (int i=0; i<3; i++) { for (int j=0; j<3; j++)  { local_inv_pose.operator()(i,j) = pose.operator()(j,i); } }
	cv::Mat inv_T = -R.t()*T;
	for (int i=0; i<3; i++) local_inv_pose.operator()(i,3) = inv_T.at<float>(i);
	for (int i=0; i<4; i++) local_inv_pose.operator()(3,i) = pose.operator()(3,i);
	return local_inv_pose;
}

void Dynamic_slam::getFrameData()  // can load use separate CPU thread(s) ?
{
	int local_verbosity_threshold = 0;
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0"<<flush;
	R.copyTo(old_R);																									// get ground truth frame to frame pose transform
	T.copyTo(old_T);
	old_pose 		= pose;
	inv_old_pose 	= inv_pose;																							// NB Confirmed this copies the data  NOT just the pointer.
	old_K			= K;
	inv_old_K		= inv_K;
	
	std::string str = txt[runcl.frame_num].c_str();																		// grab .txt file from array of files (e.g. "scene_00_0000.txt")
    char        *ch = new char [str.length()+1];
    std::strcpy (ch, str.c_str());
	cv::Mat T_alt;
    convertAhandaPovRayToStandard(ch,R,T,cameraMatrix);
																														if(verbosity>local_verbosity_threshold) {	
																															cout << "\nR=";
																															for (int i=0; i<3; i++){
																																cout <<"\n(";
																																for (int j=0; j<3; j++){ 
																																	cout << ", " << R.at<float>(i,j);
																																}cout << ")\n";
																															}cout<<endl<<flush;
																															
																															cout << "\nT=(";
																															for (int i=0; i<3; i++) cout << ", " << T.at<float>(i);
																															cout << ")\n"<<endl<<flush;
																														}
	pose 		= getPose(R, T);
	inv_pose 	= getInvPose(pose);
	/*
	for (int i=0; i<9; i++) { pose.operator()(i/3,i%3) = 1 * R.at<float>(i/3,i%3); 										cout << "\nR.at<float>("<<i/3<<","<<i%3<<") = " << R.at<float>(i/3,i%3) << ",   pose.operator()(i/3,i%3) = " << pose.operator()(i/3,i%3) ;  }
	for (int i=0; i<3; i++) pose.operator()(i,3)       = T.at<float>(i);
	for (int i=0; i<3; i++) pose.operator()(3,i)       = 0.0f;
	pose.operator()(3,3) = 1.0f;
	*/
	/*
	//for (int i=0; i<9; i++){R_dif.at<float>(i) = R.at<float>(i) - old_R.at<float>(i);   }
	//for (int i=0; i<3; i++){T_dif.at<float>(i) = T.at<float>(i) - old_T.at<float>(i);   }
	
	cout << "\n\nT.size()="<<T.size()<<flush;
	cout << "\nruncl.frame_num="<<runcl.frame_num << endl;
	cout << txt[runcl.frame_num].c_str() << flush;
	
	cout << "\n\nR_dif = ";
	for (int i=0; i<3; i++){
		cout << "\n(";
		for (int j=0; j<3; j++){
			cout << ", "<< R_dif.at<float>(i,j);
		}cout<<")";
	}cout<<flush;
	
	cout << "\nT = ";
	for (int i=0; i<3; i++){
		cout << "(";
		cout << ", "<< T.at<float>(i);
		cout<<")";
	}cout<<flush;
	
	cout << "\nold_T = ";
	for (int i=0; i<3; i++){
		cout << "(";
		cout << ", "<< old_T.at<float>(i);
		cout<<")";
	}cout<<flush;
	
	cout << "\nT_dif = ";
	for (int i=0; i<3; i++){
		cout << "(";
		cout << ", "<< T_dif.at<float>(i);
		cout<<")";
	}cout<<flush;
	
	// generate pose transform
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.1"<<flush;
	//Mat P;
	//cv::hconcat(R_dif,T_dif,P);
	*/
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.1.1"<<flush;
	/*
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){pose.operator()(i,j) = R_dif.at<float>(i,j);}
		pose.operator()(i,3) = T_dif.at<float>(i);
	}
	for (int i=0; i<3; i++) pose.operator()(3,i) = 0;
	pose.operator()(3,3) = 1.0f;
	*/
																														if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::getFrameData_chk 0.1.2"<<flush;
																															cout << "\npose = (";
																															for (int i=0; i<4; i++){
																																for (int j=0; j<4; j++){
																																	cout << ", " << pose.operator()(i,j);
																																}cout << "\n     ";
																															}cout << ")\n"<<flush;
																														}
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.2"<<flush;
	K.zeros();
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			K.operator()(i,j) = cameraMatrix.at<float>(i,j);
		}
	}K.operator()(3,3) = 1;
	generate_invK();
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.4"<<flush; // K2K
	K2K = old_K * old_pose * inv_pose * inv_K;																			// TODO  Issue, not valid for first frame, pose  should be identty, Also what would estimate SE3 do ?
	
	pose2pose = pose * inv_old_pose; 																																													// pose2pose
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.5"<<flush;
	for (int i=0; i<16; i++){ runcl.fp32_k2k[i] = K2K.operator()(i/4, i%4);   cout << "\nK2K ("<<i%4 <<","<< i/4<<") = "<< runcl.fp32_k2k[i]; }
																														// TODO insert desired pose error, to test optimisation.
	// set k2k and upload ?
	
																														if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 1,"<<flush;
																															cout << "\n\nK2K = ";
																															for (int i=0; i<4; i++){
																																cout << "\n(";
																																for (int j=0; j<4; j++){
																																	cout << ", "<< K2K.operator()(i,j);
																																}cout<<")";
																															}cout<<flush;
																															
																															cout << "\n\nruncl.fp32_k2k[i] = ";
																															for (int i=0; i<4; i++){
																																cout << "\n(";
																																for (int j=0; j<4; j++){
																																	cout << ", "<< runcl.fp32_k2k[i*4 + j];
																																}cout<<")";
																															}cout<<flush;
																															
																															cout << "\n\npose2pose = ";
																															for (int i=0; i<4; i++){
																																cout << "\n(";
																																for (int j=0; j<4; j++){
																																	cout << ", "<< pose2pose.operator()(i,j);
																																}cout<<")";
																															}cout<<flush;
																															
																															
																															cout << "\n\nK = ";
																															for (int i=0; i<4; i++){
																																cout << "\n(";
																																for (int j=0; j<4; j++){
																																	cout << ", "<< K.operator()(i,j);
																																}cout<<")";
																															}cout<<flush;
																															
																															cout << "\n\ncameraMatrix.size() = "<<cameraMatrix.size()<<flush;
																															cout << "\n\ncameraMatrix = ";
																															for (int i=0; i<3; i++){
																																cout << "\n(";
																																for (int j=0; j<3; j++){
																																	cout << ", "<< cameraMatrix.at<float>(i,j);
																																}cout<<")";
																															}cout<<endl<<flush;
																														}
																														// get ground truth depth map
																														if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 2,"<<flush;}
	int r = image.rows;
    int c = image.cols;
	depth_GT = loadDepthAhanda(depth[runcl.frame_num].string(), r,c,cameraMatrix);
																														if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 3,"<<flush;}
	
	stringstream ss;
	stringstream png_ss;
	boost::filesystem::path folder_tiff = runcl.paths.at("depth_GT");
	string type_string = runcl.checkCVtype(depth_GT.type() );
	ss << "/" << folder_tiff.filename().string() << "_" << runcl.frame_num <<"type_"<<type_string;  
	png_ss << "/" << folder_tiff.filename().string() << "_" << runcl.frame_num;
	boost::filesystem::path folder_png = folder_tiff;
	folder_tiff += ss.str();
	folder_tiff += ".tiff";
	folder_png  += "/png/";

	folder_png  += png_ss.str();
	folder_png  += ".png";
																														if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 4,"<<flush;}
	cout << "\n depth_GT.size() = " << depth_GT.size() << ",  depth_GT.type() = "<< type_string << ",  depth_GT.empty() = " <<  depth_GT.empty()   << flush;
	cout << "\n " << folder_png.string() << flush; 
	cv::imwrite(folder_png.string(), depth_GT );
	
	runcl.loadFrameData(depth_GT, K2K, pose2pose);
																														if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData finished,"<<flush;	
}

void Dynamic_slam::generate_invK(){ // TODO hack this to work here 
	int local_verbosity_threshold = 0;
	
	float fx   =  K.operator()(0,0);
	float fy   =  K.operator()(1,1);
	float skew =  K.operator()(0,1);
	float cx   =  K.operator()(0,2);
	float cy   =  K.operator()(1,2);
																							if(verbosity>local_verbosity_threshold) {
																								cout<<"\nfx="<<fx <<"\nfy="<<fy <<"\nskew="<<skew <<"\ncx="<<cx <<"\ncy= "<<cy;
																								cout << "\n\nCostVol_chk 5\n" << flush;
																							}
	///////////////////////////////////////////////////////////////////// Inverse camera intrinsic matrix, see:
	// https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/#:~:text=Inverse,lines%20from%20the%20camera%20center.
	inv_K = inv_K.zeros();
	inv_K.operator()(0,0)  = 1.0/fx;  cout<<"\n1.0/fx="<<1.0/fx;
	inv_K.operator()(1,1)  = 1.0/fy;  cout<<"\n1.0/fy="<<1.0/fy;
	inv_K.operator()(2,2)  = 1.0;
	inv_K.operator()(3,3)  = 1.0;

	inv_K.operator()(0,1)  = -skew/(fx*fy);
	inv_K.operator()(0,2)  = (cy*skew - cx*fy)/(fx*fy);
	inv_K.operator()(1,2)  = -cy/fy;
																							if(verbosity>local_verbosity_threshold) {
																								cv::Matx44f test_K = inv_K * K;
																								cout<<"\n\ntest_camera_intrinsic_matrix inversion\n";	// Verify inv_K:
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<test_K.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								std::cout << std::fixed << std::setprecision(-1);		// Inspect values in matricies ///////
																								cout<<"\n\npose\n";
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<pose.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								cout<<"\n\ninv_old_pose\n";
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<inv_old_pose.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								cout<<"\n\nK\n";
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<K.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";

																								cout<<"\n\ninv_K\n";
																								for(int i=0; i<4; i++){
																									for(int j=0; j<4; j++){
																										cout<<"\t"<< std::setw(5)<<inv_K.operator()(i,j);
																									}cout<<"\n";
																								}cout<<"\n";
																							}
}

/*
void Dynamic_slam::generate_invPose(){ // Replaced by  getInvPose(), for now.
	int local_verbosity_threshold = 1;
																						if(verbosity>local_verbosity_threshold) {
																							cout << "\nDynamic_slam::generate_invPose()" << endl << flush;
																						}
	// NB in DTAM_opencl the R and T matricies are given as arguments to the constructor.
	// invPose of keyframe wrt world coords is computed at the beginig of the cost vol.
	// For dynamic SLAM , we might nt need inv_pose if we are only interested in the pos transform previous->next frame.
																							/ *
																							*  Inverse of a transformation matrix:
																							*  http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0053
																							*
																							*   {     |   }-1       {       |        }
																							*   {  R  | t }     =   {  R^T  |-R^T .t }
																							*   {_____|___}         {_______|________}
																							*   {0 0 0| 1 }         {0  0  0|    1   }
																							*
																							* /
	cv::Matx44f poseTransform = cv::Matx44f::zeros();
	for (int i=0; i<9; i++) poseTransform.operator()(i/3,i%3) = R.at<float>(i/3,i%3);
	for (int i=0; i<3; i++) poseTransform.operator()(i,3)     = T.at<float>(i);			// why is T so large ?
	poseTransform.operator()(3,3) = 1;
	//
	
	for (int i=0; i<3; i++) { for (int j=0; j<3; j++)  { inv_old_pose.operator()(i,j) = pose.operator()(j,i); } }
	cv::Mat inv_T = -R.t()*T;
	for (int i=0; i<3; i++) inv_old_pose.operator()(i,3) = inv_T.at<float>(i);
	for (int i=0; i<4; i++) inv_old_pose.operator()(3,i) = pose.operator()(3,i);
}
*/

void Dynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] ) {	// Generates a set of 6 k2k to be used to compute the SE3 maps for the current camera intrinsic matrix.
	int local_verbosity_threshold = 0;
																						if(verbosity>local_verbosity_threshold) {
																							cout << "\nDynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] )" << endl << flush;
																						}
	// SE3 
	// Rotate 0.001 radians i.e 0.0573  degrees
	// Translate 0.001 'units' of distance 
	const float delta_theta = 0.001; //0.001;
	const float delta 	  	= 0.001; //0.001;
	const float cos_theta   = cos(delta_theta);
	const float sin_theta   = sin(delta_theta);
	
	cv::Matx44f transform[6];
	
	transform[Rx] = cv::Matx44f(1,         0,          0,          0,				0,         cos_theta, -sin_theta, 0,				0,           sin_theta, cos_theta, 0,				0, 0, 0, 1);
	transform[Ry] = cv::Matx44f(cos_theta, 0,          sin_theta,  0,				0,         1,         0,          0,				-sin_theta,  0,         cos_theta, 0,				0, 0, 0, 1);
	transform[Rz] = cv::Matx44f(cos_theta, -sin_theta, 0,          0, 				sin_theta, cos_theta, 0,          0,				0,           0,         1,         0,				0, 0, 0, 1);
	
	transform[Tx] = cv::Matx44f(1,0,0,delta, 	0,1,0,0,		0,0,1,0,		0,0,0,1);
	transform[Ty] = cv::Matx44f(1,0,0,0, 		0,1,0,delta,	0,0,1,0,		0,0,0,1);
	transform[Tz] = cv::Matx44f(1,0,0,0, 		0,1,0,0,		0,0,1,delta,	0,0,0,1);
	
	cv::Matx44f cam2cam[6];
	for (int i=0; i<6; i++) { 
		cam2cam[i] = K*transform[i]*inv_K; 
		if(verbosity>local_verbosity_threshold) { cout << "\ntransform["<<i<<"]=\n"<<transform[i]<<endl<<flush; }
	}
	
	for (int i=0; i<6; i++) {
		cam2cam[i] = K * transform[i] *  inv_K;
		if(verbosity>local_verbosity_threshold) { cout << "\ncam2cam["<<i<<"]=\n"<<cam2cam[i]<<endl<<flush; }
		
		for (uint row=0; row<4; row++) {
			for (uint col=0; col<4; col++){
				_SE3_k2k[i*16 + row*4 + col] 	= cam2cam[i].operator()(row,col);
			}
		}
	}
																						if(verbosity>local_verbosity_threshold) {
																							cout << setprecision(9);
																							for (int i=0; i<6; i++) {
																								cout << "\n _SE3_k2k ["<<i<<"*16 + row*4 + col]=\n";
																								for (int row=0; row<4; row++) {
																									for (int col=0; col<4; col++){
																										cout << _SE3_k2k[i*16 + row*4 + col] <<"\t  ";
																									}cout<<endl;
																								}cout<<endl;
																							}
																						
																							// Chk k2k makes sense for each SE3 DoF
																							// Image size 640 * 480, from K as loaded from .json file.
																							// Homogeneous image coords : (col, row, depth, w), so ater transformation divide by w to find new (col, row, depth, 1).
																							// NB if there are points at infinite distance, then w=0. 
																							// For this reason we store inverse depth, and use a minimum depth cut off, to avoid points at the optical centre of the camera.
																							// See __kernel void compute_param_maps(..) , or hand calculate the elements of the matrix multiplication.
																							cv::Matx14f topleft = {10,10,1,1},  topright = {630,10,1,1}, centre = {320,240,1,1}, bottomleft = {10,470,1,1}, bottomright = {630,470,1,1} , result;	// (column, row, w, 1/depth)
																							cv::Matx44f identity = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};
																							
																							cout << "\n topleft * 	identity = " << topleft * 		identity << flush;
																							cout << "\n topright * 	identity = " << topright * 		identity << flush;
																							cout << "\n bottomleft *  identity = " << bottomleft * 	identity << flush;
																							cout << "\n bottomright * identity = " << bottomright * 	identity << flush;
																							/* Incorrect calculation. Rather see __kernel void compute_param_maps(..)
																							for (int tx = 0; tx<6; tx++){
																								cout << "\n\n transform["<<tx<<"]:";
																								result = topleft * transform[tx]; 		cout << "\n topleft * transform[tx] = "  	<< result << ", \t  (" << result.operator()(0)/result.operator()(3) <<","<< result.operator()(0)/result.operator()(3) <<" ) "<< flush;
																								result = topright * transform[tx]; 		cout << "\n topright * transform[tx] = " 	<< result << ", \t  (" << result.operator()(0)/result.operator()(3) <<","<< result.operator()(0)/result.operator()(3) <<" ) "<< flush;
																								result = centre * transform[tx]; 		cout << "\n centre * transform[tx] = "   	<< result << ", \t  (" << result.operator()(0)/result.operator()(3) <<","<< result.operator()(0)/result.operator()(3) <<" ) "<< flush;
																								result = bottomleft * transform[tx]; 	cout << "\n bottomleft * transform[tx] = " 	<< result << ", \t  (" << result.operator()(0)/result.operator()(3) <<","<< result.operator()(0)/result.operator()(3) <<" ) "<< flush;
																								result = bottomright * transform[tx]; 	cout << "\n bottomright * transform[tx] = " << result << ", \t  (" << result.operator()(0)/result.operator()(3) <<","<< result.operator()(0)/result.operator()(3) <<" ) "<< flush;
																							}cout << setprecision(-1)<< flush;
																							*/
																						}
																						if(verbosity>local_verbosity_threshold) {
																							cout << "\n\nDynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] )   finished" << endl << flush;
																						}
}


void Dynamic_slam::estimateSO3()
{
// # Get 1st & 2nd order gradients of SO3 wrt predicted pose.
//


// # Predict 1st least squares step of SO3
//
	
}

void Dynamic_slam::estimateSE3(){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::estimateSE3()_chk 0\n" << flush;}
// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
// 
	runcl.estimateSE3(0,8);//(uint start=0, uint stop=8);


// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
//


// # Pass prediction to lower layers. Does it fit better ?
//


// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
//
	
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::estimateSE3()_chk 1  Finished\n" << flush;}	
}

void Dynamic_slam::estimateCalibration()
{
// # Get 1st & 2nd order gradients wrt calibration parameters. 
//


// # Take one dammped least squares step of calibration.
//

}

void Dynamic_slam::buildDepthCostVol(){
// # Build depth cost vol on current image, using image array[6] in MipMap buffer, plus RelVelMap, 
// with current camera params & DepthMap if bootstrapping, otherwise with params for each frame.
// NB 2*(1+7) = 14 layers on MipMap DepthCostVol: for model & pyramid, ID cetntre layer plus 7 samples, i.e. centre +&- 3 layers.
	
	
	
// Select naive depth map
// See CostVol::updateCost(..), RunCL::calcCostVol(..) &  __kernel void BuildCostVolume2
	
	
	
}

// ## Regularize Maps : AbsDepth, GradDepth, SurfNormal, RelVel, 
void Dynamic_slam::SpatialCostFns(){
// # Spatial cost functions
// see CostVol::updateQD(..), RunCL::updateQD(..) & __kernel void UpdateQD(..)
	
}

void Dynamic_slam::ParsimonyCostFns(){
// # Parsimony cost functions : NB Bin sort pixels to find non-spatial neighbours
// see SIFS for priors & Morphogenesis for BinSort
	
}

void Dynamic_slam::ExhaustiveSearch(){
// # Update A : exhaustive search on cost vol with cost fns -> update maps.
// see CostVol::updateA(..), RunCL::updateA(..) & __kernel void UpdateA2(..)

}



void Dynamic_slam::getResult()
{
	
};
    

/////////////////////////////////////////////////////////////////
    
/*void Dynamic_slam::getFrame() {
// # load next image to buffer NB load at position [log_2 index]
// See CostVol::updateCost(..) & RunCL::calcCostVol(..) 

// # convert colour space
//

// # loop kernel for reduction n-times on MipMap
//

// # Get 1st & 2nd order image gradients of MipMap
// see CostVol::cacheGValues(), RunCL::cacheGValue2 & __kernel void CacheG3
}*/


/*void Dynamic_slam::predictFrame()
// # Predict expected camera motion from previous SE3 vel & accel (both zero initially)
//
*/

/*void Dynamic_slam::estimateSO3() {
// # Get 1st & 2nd order gradients of SO3 wrt predicted pose.
//


// # Predict 1st least squares step of SO3
//}
*/

/*void Dynamic_slam::estimateSE3()
{
// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
//


// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
//


// # Pass prediction to lower layers. Does it fit better ?
//


// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
//
}
*/

//////////////////////////////////////////////////////////////////
/*void Dynamic_slam::estimateCalibration()
{
// # Get 1st & 2nd order gradients wrt calibration parameters. 
//


// # Take one dammped least squares step of calibration.
//

}
*/
//////////////////////////////////////////////////////////////////

/*void Dynamic_slam::buildDepthCostVol(){
// # Build depth cost vol on current image, using image array[6] in MipMap buffer, plus RelVelMap, 
// with current camera params & DepthMap if bootstrapping, otherwise with params for each frame.
// NB 2*(1+7) = 14 layers on MipMap DepthCostVol: for model & pyramid, ID cetntre layer plus 7 samples, i.e. centre +&- 3 layers.
// Select naive depth map
// See CostVol::updateCost(..), RunCL::calcCostVol(..) &  __kernel void BuildCostVolume2
}
*/

// ## Regularize Maps : AbsDepth, GradDepth, SurfNormal, RelVel, 
/*void Dynamic_slam::SpatialCostFns(){
// # Spatial cost functions
// see CostVol::updateQD(..), RunCL::updateQD(..) & __kernel void UpdateQD(..)
}
*/

/*void Dynamic_slam::ParsimonyCostFns(){
// # Parsimony cost functions : NB Bin sort pixels to find non-spatial neighbours
// see SIFS for priors & Morphogenesis for BinSort
}
*/

/*void Dynamic_slam::ExhaustiveSearch(){
// # Update A : exhaustive search on cost vol with cost fns -> update maps.
// see CostVol::updateA(..), RunCL::updateA(..) & __kernel void UpdateA2(..)
}
*/


