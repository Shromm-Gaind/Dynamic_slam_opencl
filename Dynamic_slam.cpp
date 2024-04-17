#include "Dynamic_slam.h"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;


Dynamic_slam::~Dynamic_slam(){
	
};

Dynamic_slam::Dynamic_slam( Json::Value obj_ ): runcl(obj_ ) {
	int local_verbosity_threshold = -2;
	obj = obj_;
	//resultsMat = cv::Mat();
	//runcl.resultsMat = resultsMat;
	verbosity 					= obj["verbosity"].asInt();
	runcl.dataset_frame_num 	= obj["data_file_offset"].asUInt();
	invert_GT_depth  			= obj["invert_GT_depth"].asBool();
	
	SE3_start_layer 			= obj["SE3_start_layer"].asInt();
    SE3_stop_layer 				= obj["SE3_stop_layer"].asInt();
	SE_iter_per_layer 			= obj["SE_iter_per_layer"].asInt();
    SE_iter 					= obj["SE_iter"].asInt();
	SE_factor					= obj["SE_factor"].asFloat();

	for (int layer=0; layer<MAX_LAYERS; layer++){for (int chan=0; chan<3; chan++)	SE3_Rho_sq_threshold[layer][chan]  	= obj["SE3_Rho_sq_threshold"][layer][chan].asFloat();  }
	for (int se3=0; se3<8; se3++)													SE3_update_dof_weights[se3] 		= obj["SE3_update_dof_weights"][se3].asFloat();
    for (int layer=0; layer<MAX_LAYERS; layer++) 									SE3_update_layer_weights[layer] 	= obj["SE3_float update_layer_weights"][layer].asFloat();

																																			if(verbosity>local_verbosity_threshold-4) {cout << "\n Dynamic_slam::Dynamic_slam_chk 0,  SE3_Rho_sq_threshold[i][j] = ";
																																				for (int i=0; i<5; i++){cout << "( "; for (int j=0; j<3; j++) {cout << ", ["<<i<<"]["<<j<<"]" << SE3_Rho_sq_threshold[i][j]; }   cout << " )";}
																																				cout << ",\t SE_factor = "<<SE_factor;
																																				cout << endl << flush;
																																			}
	stringstream ss0;
	ss0 << obj["data_path"].asString() << obj["data_file"].asString();
	rootpath = ss0.str();
	root = rootpath;
	
	if ( exists(root)==false )		{ cout << "Data folder "<< ss0.str()  <<" does not exist.\n" <<flush; exit(0); }
	if ( is_directory(root)==false ){ cout << "Data folder "<< ss0.str()  <<" is not a folder.\n"<<flush; exit(0); }
	if ( empty(root)==true )		{ cout << "Data folder "<< ss0.str()  <<" is empty.\n"		 <<flush; exit(0); }
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 1\n" << flush;
	get_all(root, ".txt",   txt);																											// Get lists of files. Gathers all filepaths with each suffix, into c++ vectors.
	get_all(root, ".png",   png);
	get_all(root, ".depth", depth);
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 2\n" << flush;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam(): "<< png.size()  <<" .png images found in data folder.\t"<<"png[runcl.dataset_frame_num].string()="<< png[runcl.dataset_frame_num].string()  <<flush;
	runcl.baseImage 	= imread(png[runcl.dataset_frame_num].string());																	// Set image params, ref for dimensions and data type.
																																			if (verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam_chk 3: runcl.baseImage.size() = "<< runcl.baseImage.size() \
																																				<<" runcl.baseImage.type() = " << runcl.baseImage.type() << "\t"<< runcl.checkCVtype(runcl.baseImage.type()) <<flush;
																																			if(verbosity>1) { imshow("runcl.baseImage",runcl.baseImage); cv::waitKey(-1); }
	runcl.initialize();
	runcl.allocatemem();
																																			if (verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam_chk 4: runcl.baseImage.size() = "<< runcl.baseImage.size() \
																																				<<" runcl.baseImage.type() = " << runcl.baseImage.type() << "\t"<< runcl.checkCVtype(runcl.baseImage.type()) <<flush;
	initialize_camera();																													if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 5\n" << flush;
	//generate_SE3_k2k( SE3_k2k );																											if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 6\n" << flush;
	//runcl.precom_param_maps( SE3_k2k );
	//initialize_resultsMat();
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 7 finished\n" << flush;
};

void Dynamic_slam::initialize_resultsMat(){	// need to take layer 2, or read it from .json .
	int local_verbosity_threshold = -2;																										if(verbosity>local_verbosity_threshold) cout << "\n\n Dynamic_slam::initialize_resultsMat()_chk 1" << flush;
	uint reduction 	= obj["sample_layer"].asUInt();
	uint SE_iter 	= obj["SE_iter"].asUInt();																								if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::initialize_resultsMat()_chk 2" << flush;
	int rows 		= 7 * ( runcl.MipMap[reduction*8 + MiM_READ_ROWS] +  runcl.mm_margin );
	int cols 		= SE_iter * ( runcl.MipMap[reduction*8 + MiM_READ_COLS] +  runcl.mm_margin );											if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::initialize_resultsMat()_chk 3: rows = "<<rows<<", cols = "<<cols<< flush;
	runcl.resultsMat = cv::Mat::zeros ( rows, cols , CV_8UC4);																				if(verbosity>local_verbosity_threshold) cout << ",  runcl.resultsMat.size() = "<< runcl.resultsMat.size() << flush;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::initialize_resultsMat()_chk  finished\n" << flush;
}//   Dynamic_slam::initialize_resultsMat()_chk 3: rows = 882, cols = 1660,  runcl.resultsMat.size() = [1660 x 882]

void Dynamic_slam::initialize_camera(){
	int local_verbosity_threshold = 2;
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_chk 0:" <<flush;}
	K = K.zeros();																															// NB In DTAM_opencl, "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	K.operator()(3,3) = 1;
	for (int i=0; i<9; i++){K.operator()(i/3,i%3) = obj["cameraMatrix"][i].asFloat(); }
	old_K		= K;
	generate_invK();
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
	//generate_invK();
	R 				= cv::Mat::eye(3,3 , CV_32FC1);																									// intialize ground truth extrinsic data, NB Mat (int rows, int cols, int type)
	T 				= cv::Mat::zeros(3,1 , CV_32FC1);
	//SO3_pose2pose	= Matx33f_eye;
	pose2pose		= Matx44f_eye;
	K2K				= Matx44f_eye;
	//for (int i=0; i<16; i++){pose2pose.operator()(i/4,i%4)	=0;} 
	//for (int i=0; i< 4; i++){pose2pose.operator()(i,i)		=1;}
	//for (int i=0; i<16; i++){K2K.operator()(i/4,i%4) 			= pose2pose.operator()(i/4,i%4);}
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_chk 1:" <<flush;
																																				for (int i=0; i<9; i++){cout << "\n R.at<float>("<<i<<")="<< R.at<float>(i)<< flush ;  }
																																			}
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_chk 2:" <<flush;}
	// TODO Also initialize any lens distorsion, vignetting. etc
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_chk 3:" <<flush;}
	getFrameData();																															// Loads GT depth of the new frame. NB depends on image.size from getFrame().
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_chk 4:" <<flush;}
	K_start 					= K_GT; 																										// NB The same frame will be loaded to the opposite imgmem, on the first iteration of Dynamic_slam::nextFrame()
	inv_K_start 				= inv_K_GT; 
	pose_start 					= pose_GT;
	inv_pose_start 				= inv_pose_GT;
	K2K_GT 						= Matx44f_eye; 																								//  = cv::Matx44f::eye();
	K2K_start 					= Matx44f_eye;
	pose2pose_GT 				= Matx44f_eye;
	pose2pose_start 			= Matx44f_eye;
	pose2pose_accumulated 		= Matx44f_eye;
	pose2pose_accumulated_GT 	= Matx44f_eye;
																																			if (verbosity>local_verbosity_threshold){ cout << "\nDynamic_slam::initialize_camera_chk 5:" <<flush;
																																				cout << "\nDynamic_slam::initialize_camera: pose2pose_accumulated = "; for (int i=0; i<16; i++) {cout << ", " << pose2pose_accumulated.operator()(i/4,i%4); }
																																			}cout << flush;
	generate_SE3_k2k( SE3_k2k );
	runcl.precom_param_maps( SE3_k2k );
	getFrame();																																// Causes the first frame to be loaded into first imgmem, and prepared. 
}

int Dynamic_slam::nextFrame() {
	int local_verbosity_threshold = -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::nextFrame_chk 0,  runcl.dataset_frame_num="<<runcl.dataset_frame_num<<" \n" << flush; //  runcl.frame_bool_idx="<<runcl.frame_bool_idx<<"
	predictFrame();					// updates pose2pose for next frame in cost volume.			//  Dynamic_slam::getFrameData_chk 0.  runcl.dataset_frame_num = 0
	getFrameData();					// Loads GT depth of the new frame. NB depends on image.size from getFrame().
	use_GT_pose();
	getFrame();
	if(obj["Artif_pose_err_factor"].asFloat() != 0.0 ) artificial_pose_error(); else cout << "\nArtif_pose_err_factor = "<<obj["Artif_pose_err_factor"].asFloat() << flush;
//	estimateSO3();
	//artificial_pose_error();
	//estimateSE3(); 					// own thread ? num iter ?
	estimateSE3_LK(); 					// own thread ? num iter ?

	//estimateCalibration(); 		// own thread, one iter.
	report_GT_pose_error();
	display_frame_resluts();
	////////////////////////////////// Test kernels

	//runcl.atomic_test1();

	////////////////////////////////// Parallax depth mapping
	
	updateDepthCostVol();													// Update cost vol with the new frame, and repeat optimization of the depth map.
																			// NB Cost vol needs to be initialized on a particular keyframe.
																			// A previous depth map can be transfered, and the updated depth map after each frame, can be used to track the next frame.
	runcl.costvol_frame_num++;
	runcl.dataset_frame_num++;
	
	return(0);						// NB option to return an error that stops the main loop.
};

void Dynamic_slam::optimize_depth(){
	bool doneOptimizing;
	int  opt_count				= 0;
	int  max_opt_count			= obj["max_opt_count"].asInt();
	int  max_inner_opt_count 	= obj["max_inner_opt_count"].asInt();
	
	runcl.QD_count				= 0;
	runcl.A_count				= 0;
	runcl.G_count				= 0;
	runcl.initialize_fp32_params();											// reset params for update QD & A
	
	cacheGValues();															// initialize "keyframe_g1mem"" map of image edges.
	do{ 
		for (int i = 0; i < max_inner_opt_count; i++) updateQD();			// Optimize Q, D   (primal-dual)		/ *5* /
		doneOptimizing = updateA();											// Optimize A      (pointwise exhaustive search)
		opt_count ++;
	} while (!doneOptimizing && (opt_count<max_opt_count));
	
	//////////////////////////////////
	int outer_iter = obj["regularizer_outer_iter"].asInt();
	int inner_iter = obj["regularizer_inner_iter"].asInt();
	
	for (int i=0; i<outer_iter ; i++){
		for (int j=0; j<inner_iter ; j++){
			SpatialCostFns();
			ParsimonyCostFns();
		}
		ExhaustiveSearch();
	}
}

void Dynamic_slam::report_GT_pose_error(){
	int local_verbosity_threshold = -2;
	//static void PToLie(InputArray _P, OutputArray Lie); //
	PToLie(pose2pose_accumulated_GT, pose2pose_accumulated_GT_algebra); 																	// pose2pose_accumulated_GT_algebra 		= SE3_Algebra(pose2pose_accumulated_GT);
	PToLie(pose2pose_accumulated  ,  pose2pose_accumulated_algebra); 																		// pose2pose_accumulated_algebra 			= SE3_Algebra(pose2pose_accumulated);
	//static Mat LieSub(Mat A, Mat B);
	pose2pose_accumulated_error_algebra = LieSub(pose2pose_accumulated_algebra, pose2pose_accumulated_GT_algebra); 							//pose2pose_accumulated_error_algebra 	= pose2pose_accumulated_algebra - pose2pose_accumulated_GT_algebra;

	PToLie(pose2pose_GT  ,  pose2pose_GT_algebra); 																							//pose2pose_GT_algebra 					= SE3_Algebra(pose2pose_GT);
	PToLie(pose2pose     ,  pose2pose_algebra); 																							//pose2pose_algebra 						= SE3_Algebra(pose2pose);
	pose2pose_error_algebra = LieSub(pose2pose_algebra, pose2pose_GT_algebra); 																//pose2pose_error_algebra 				= pose2pose_algebra - pose2pose_GT_algebra;

																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::report_GT_error_chk 0\n" << flush;
																																				cout << "\npose2pose_accumulated, runcl.frame_num = " << runcl.dataset_frame_num ;
																																				cout << "\npose2pose_accumulated_GT = "; for (int i=0; i<16; i++) cout << ", " << pose2pose_accumulated_GT.operator()(i/4,i%4);
																																				cout << "\npose2pose_accumulated    = "; for (int i=0; i<16; i++) cout << ", " << pose2pose_accumulated.operator()(i/4,i%4);
																																				cout << "\npose2pose_accumulated_error_algebra = "; for (int i=0; i<6; i++) cout << ", " << pose2pose_accumulated_error_algebra.operator()(i);
																																				cout << endl;
																																				cout << "\npose2pose_GT = "; for (int i=0; i<16; i++) cout << ", " << pose2pose_GT.operator()(i/4,i%4);
																																				cout << "\npose2pose    = "; for (int i=0; i<16; i++) cout << ", " << pose2pose.operator()(i/4,i%4);
																																				//
																																				cout << "\npose2pose_GT_algebra = "; 	for (int i=0; i<6; i++) cout << ", " << pose2pose_GT_algebra.operator()(i);
																																				cout << "\npose2pose_algebra = "; 		for (int i=0; i<6; i++) cout << ", " << pose2pose_algebra.operator()(i);
																																				cout << "\npose2pose_error_algebra = "; for (int i=0; i<6; i++) cout << ", " << pose2pose_error_algebra.operator()(i);
																																				cout << endl;
																																				cout << flush;
																																			}
}

void Dynamic_slam::display_frame_resluts(){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::display_frame_resluts_chk 0\n" << flush;}
	stringstream ss; 
	ss << "_p2p_error_" ;
	for (int i=0; i<6; i++) ss << "," << pose2pose_error_algebra.operator()(i);
	ss << "_cumulative_error_" ;
	for (int i=0; i<6; i++) ss << "," << pose2pose_accumulated_error_algebra.operator()(i);
	runcl.tracking_result( ss.str() );
}

/*
void Dynamic_slam::artificial_SO3_pose_error(){
	int local_verbosity_threshold =-2;
																																			if(verbosity>local_verbosity_threshold) {cout << "\nDynamic_slam::artificial_SO3_pose_error()_chk_0" << flush;}
																																			if(verbosity>local_verbosity_threshold) {cout << "\ntransform[2]       = "; for (int i=0; i<16; i++) cout << ", " << transform[2].operator()(i/4,i%4);}
	Matx44f poseStep = transform[2];
																																			if(verbosity>local_verbosity_threshold) {cout << "\nposeStep           = "; for (int i=0; i<16; i++) cout << ", " << poseStep.operator()(i/4,i%4);}
	Matx33f SO3_poseStep;
	for (int i=0; i<9; i++){ SO3_poseStep.operator()(i/3,i%3) =  poseStep.operator()(i/3,i%3); }
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\nDynamic_slam::artificial_SO3_pose_error()_chk_1" << flush;}
																																			if(verbosity>local_verbosity_threshold) {cout << "\nSO3_poseStep       = "; for (int i=0; i<9; i++) cout << ", " << SO3_poseStep.operator()(i/3,i%3);}
	Matx33f big_step = SO3_poseStep; for (int i = 0; i<1; i++) big_step = big_step * SO3_poseStep;											// Iterations of Max mul.
																																			if(verbosity>local_verbosity_threshold) {cout << "\nbig_step           = "; for (int i=0; i<9; i++) cout << ", " << big_step.operator()(i/3,i%3); }
	SO3_pose2pose = SO3_pose2pose * big_step;																								// Add error of one step in SO3.
																																			if(verbosity>local_verbosity_threshold) {cout << "\nnew_SO3_pose2pose  = "; for (int i=0; i<9; i++) cout << ", " << SO3_pose2pose.operator()(i/3,i%3);}
	for (int i=0; i<9; i++){ runcl.fp32_so3_k2k[i] = SO3_pose2pose.operator()(i/3, i%3);}   												if(verbosity>local_verbosity_threshold) {cout << "\nruncl.fp32_so3_k2k = "; for (int i=0; i<9; i++) cout << ", " << runcl.fp32_so3_k2k[i]; }
}
*/

void Dynamic_slam::artificial_pose_error(){
	int local_verbosity_threshold =-2;
	Matx61f pose_step_algebra;
	for (int SE3=0; SE3<6; SE3++)  pose_step_algebra.operator()(SE3,0) = obj["Artif_pose_err_algebra"][SE3].asFloat();
	Matx44f poseStep = LieToP_Matx(pose_step_algebra);																						//= SE3_Matx44f(pose_step_algebra);
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\nposeStep = ";for(int row=0;row<4;row++){ cout<<"  \n"; for(int col=0; col<4;col++)cout<<poseStep.operator()(row,col)<<", \t";}
																																				cout << "\npose_step_algebra = ";for(int col=0; col<6;col++)cout<<pose_step_algebra.operator()(col)<<", ";
																																				for (int SE3=0; SE3<6; SE3++) cout << "\nArtif_pose_err_algebra = "<<obj["Artif_pose_err_algebra"][SE3].asFloat();
																																				cout << endl << flush;
																																			}
	/*
################################
	uint axis = obj["Artif_pose_err_axis"].asUInt() ;
	Matx44f poseStep = transform[ axis ];
																																			if(verbosity>local_verbosity_threshold) {cout << "\nDynamic_slam::artificial_pose_error()_chk_0" << flush;}
																																			if(verbosity>local_verbosity_threshold) {cout << "\nposeStep = "; for (int i=0; i<16; i++) cout << ", \t" << poseStep.operator()(i/4,i%4);}
	float factor = obj["Artif_pose_err_factor"].asFloat();
	//if (axis>2) factor *= obj["min_depth"].asFloat();
	Matx44f big_step = poseStep; for (int i = 0; i<factor; i++) big_step = big_step * poseStep;												// Iterations of Max mul.
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\nbig_step = "; 	for (int i=0; i<16; i++) cout << ", \t" << big_step.operator()(i/4,i%4);
																																				cout << "\npose2pose = "; 	for (int i=0; i<16; i++) cout << ", \t" << pose2pose.operator()(i/4,i%4);
																																				cout << "\nold_K2K = "; 	for (int i=0; i<16; i++) cout << ", \t" << K2K.operator()(i/4,i%4);
																																			}
	*/
																																			if(verbosity>local_verbosity_threshold){
																																				Matx61f pose2pose_SE3_Algebra; // = SE3_Algebra(pose2pose);		//SE3_Algebra(const Matx44f& SE3_Matx)
																																				PToLie(pose2pose, pose2pose_SE3_Algebra);
																																				cout << "\nTrue_pose2pose_SE3_Algebra = "; 	for (int i=0; i<6; i++) cout << ", \t" << pose2pose_SE3_Algebra.operator()(i,0);  		//[i][0];
																																			}
	pose2pose = pose2pose * poseStep;	//big_step;
																																			if(verbosity>local_verbosity_threshold){
																																				Matx61f pose2pose_SE3_Algebra; // = SE3_Algebra(pose2pose);		//SE3_Algebra(const Matx44f& SE3_Matx)
																																				PToLie(pose2pose, pose2pose_SE3_Algebra);
																																				cout << "\tStart_pose2pose_SE3_Algebra = "; 	for (int i=0; i<6; i++) cout << ", \t" << pose2pose_SE3_Algebra.operator()(i,0);  	//[i][0];
																																				cout << endl << flush;
																																			}
	K2K = old_K * pose2pose * inv_K;																										// Add error of one step in the 2nd SE3 DoF.
																																			if(verbosity>local_verbosity_threshold){ cout << "\nnew_K2K = "; for (int i=0; i<16; i++) cout << ", \t" << K2K.operator()(i/4,i%4);}
	
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\nold runcl.fp32_k2k = "; 	for (int i=0; i<16; i++) cout << ", \t" << runcl.fp32_k2k[i];
																																			}

																																			if(verbosity>local_verbosity_threshold){ cout << "\nnew runcl.fp32_k2k = "; }
	for (int i=0; i<16; i++){ runcl.fp32_k2k[i] = K2K.operator()(i/4, i%4);   																if(verbosity>local_verbosity_threshold)  cout << ",\t"<< runcl.fp32_k2k[i]; }
	//runcl.loadFrameData(depth_GT, K2K, pose2pose);																						// NB runcl.fp32_k2k is loaded to k2kbuf by RunCL::estimateSE3(..)
																																			if(verbosity>local_verbosity_threshold) {cout << "\nDynamic_slam::artificial_SO3_pose_error()_finish ##############################################" << flush;}
}

void Dynamic_slam::predictFrame(){
	int local_verbosity_threshold = -2;
	//for (int i=0; i<16; i++)  pose2pose.operator()(i/4,i%4) =     ;   
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num  << flush;
																																				cout << "\nOld K2K        		= ";  for (int i=0; i<16; i++) cout << ", \t" << K2K.operator()(i/4,i%4);
																																				cout << "\nOld pose2pose        = ";  for (int i=0; i<16; i++) cout << ", \t" << pose2pose.operator()(i/4,i%4); cout << endl << flush;
																																			}
	old_K 					= keyframe_K;			//= K;
	inv_old_K				= keyframe_inv_K;		//= inv_K;
	old_pose				= keyframe_pose;		//= pose;
	inv_old_pose			= keyframe_inv_pose;	//= inv_pose;
	keyframe_pose2pose		= pose2pose;
	pose2pose_algebra_2		= pose2pose_algebra_1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 0.1"<< flush;}
	PToLie(pose2pose, pose2pose_algebra_1);			//pose2pose_algebra_1		= SE3_Algebra(pose2pose);
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 0.2"<< flush;}
	Matx61f p2p_alg = LieSub(pose2pose_algebra_1, pose2pose_algebra_2);
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 0.3"<< flush;}
	if (runcl.costvol_frame_num==0){ pose2pose_algebra_0	= 								(runcl.dataset_frame_num > 2)*(p2p_alg );} /*pose2pose_algebra_1 - pose2pose_algebra_2*/
	else{ 							 pose2pose_algebra_0	= LieAdd(pose2pose_algebra_1, 	(runcl.dataset_frame_num > 2)*(p2p_alg));} //pose2pose_algebra_1 + (runcl.dataset_frame_num > 2)*(p2p_alg) ;} //+ (runcl.frame_num > 2)*0.5*(pose2pose_algebra_1 - pose2pose_algebra_2);	// Only use accel if there are enough previous frames.

																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 0.4"<< flush;}
	pose2pose = LieToP_Matx(pose2pose_algebra_0 ); 				//pose2pose = SE3_Matx44f(pose2pose_algebra_0);
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 0.5"<< flush;}
	K2K = old_K * pose2pose * inv_K;
	
	keyframe_K2K = K * pose * keyframe_inv_pose * inv_old_K;   // TODO fix this ?  Pose should reflect intertia, i.e. expect the same step again.
	
	for (int i=0; i<16; i++){ runcl.fp32_k2k[i] = K2K.operator()(i/4, i%4); }
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 1\n" << flush;
																																				cout << "\nruncl.dataset_frame_num  = " << runcl.dataset_frame_num;
																																				cout << "\nruncl.costvol_frame_num  = " << runcl.costvol_frame_num;
																																				cout << "\npose2pose_algebra_2      = ";  for (int i=0; i< 6; i++) cout << ", \t" << pose2pose_algebra_2.operator()(i,0);
																																				cout << "\npose2pose_algebra_1      = ";  for (int i=0; i< 6; i++) cout << ", \t" << pose2pose_algebra_1.operator()(i,0);
																																				cout << "\npose2pose_algebra_0      = ";  for (int i=0; i< 6; i++) cout << ", \t" << pose2pose_algebra_0.operator()(i,0);
																																				cout << "\nNew pose2pose            = ";  for (int i=0; i<16; i++) cout << ", \t" << pose2pose.operator()(i/4,i%4);		/*New_*/
																																				cout << "\nNew_K2K        		    = ";  for (int i=0; i<16; i++) cout << ", \t" << K2K.operator()(i/4,i%4);				/*New_*/
																																				cout << "\nkeyframe_K2K				= ";  for (int i=0; i<16; i++) cout << ", \t" << keyframe_K2K.operator()(i,0);
																																				cout << endl << flush;
																																			}
	// kernel update DepthMap with RelVelMap
	
	// kernel predict new frame 
};

void Dynamic_slam::getFrame() { // can load use separate CPU thread(s) ?  // NB also need to change type CV_8UC3 -> CV_16FC3
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrame_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num  << flush;
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
	image = imread(png[runcl.dataset_frame_num].string());																					if(verbosity>local_verbosity_threshold){
																																				cout << "\n Dynamic_slam::getFrame_chk 0.5, Image file = " << png[runcl.dataset_frame_num].string() << "\t" << flush;
																																			}
																																			if (image.type()!= runcl.baseImage.type() || image.size()!=runcl.baseImage.size() ) {
																																				cout<< "\n\nError: Dynamic_slam::getFrame(), runcl.dataset_frame_num = " << runcl.dataset_frame_num << " : missmatched. runcl.baseImage.size()="<<runcl.baseImage.size()<<\
																																				", image.size()="<<image.size()<<", runcl.baseImage.type()="<<runcl.baseImage.type()<<", image.type()="<<image.type()<<"\n\n"<<flush;
																																				exit(0);
																																			}
																																			//image.convertTo(image, CV_16FC3, 1.0/256, 0.0); // NB cv_16FC3 is preferable, for faster half precision processing on AMD, Intel & ARM GPUs. 
	runcl.loadFrame( image );																												// NB Nvidia GeForce have 'Tensor Compute" FP16, accessible by PTX. AMD have RDNA and CDNA. These need PTX/assembly code and may use BF16 instead of FP16.
																																			// load a basic image in CV_8UC3, then convert on GPU to 'half'
	runcl.cvt_color_space( );
	runcl.blur_image();
	runcl.mipmap_linear();																													// (uint num_reductions, uint gaussian_size)// TODO set these as params in conf.json
	runcl.img_variance();
	runcl.img_gradients();
																																			// # Get 1st & 2nd order image gradients of MipMap
																																			// see CostVol::cacheGValues(), RunCL::cacheGValue2 & __kernel void CacheG3
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrame_chk 1  Finished\n" << flush;}
}

cv::Matx44f Dynamic_slam::getPose(Mat R, Mat T){																							// Mat R, Mat T, Matx44f& pose  // NB Matx::operator()() does not copy, but creates a submatrix. => would be updated when R & T are updated.
	int local_verbosity_threshold = 1;
	cv::Matx44f pose;
	for (int i=0; i<9; i++) {pose.operator()(i/3,i%3) = 1 * R.at<float>(i/3,i%3); 															if(verbosity>local_verbosity_threshold) cout << "\nR.at<float>("<<i/3<<","<<i%3<<") = " << R.at<float>(i/3,i%3) << ",   pose.operator()(i/3,i%3) = " << pose.operator()(i/3,i%3) ;  }
	for (int i=0; i<3; i++) pose.operator()(i,3)      = T.at<float>(i);
	for (int i=0; i<3; i++) pose.operator()(3,i)      = 0.0f;
	pose.operator()(3,3) = 1.0f;
	return pose;
}

cv::Matx44f Dynamic_slam::getInvPose(cv::Matx44f pose) {	// Matx44f pose, Matx44f& inv_pose
	cv::Matx44f local_inv_pose;
	cv::Matx33f local_rotation;
	cv::Matx31f local_translation;
	cv::Matx31f inv_local_translation;
	
	for (int i=0; i<3; i++) { for (int j=0; j<3; j++)	{    local_inv_pose.operator()(i,j) = pose.operator()(j,i); } }
	for (int i=0; i<3; i++) { for (int j=0; j<3; j++)	{    local_rotation.operator()(i,j) = pose.operator()(i,j); } }
	for (int i=0; i<3; i++) 							{ local_translation.operator()(i,0) = pose.operator()(i,3); }
	
	inv_local_translation = - local_rotation.t() * local_translation;
	for (int i=0; i<3; i++) local_inv_pose.operator()(i,3) = inv_local_translation.operator()(i,0);
	for (int i=0; i<4; i++) local_inv_pose.operator()(3,i) =                  pose.operator()(3,i);
	
	cout << "\n\nlocal_translation =";		for (int i=0; i< 3; i++) cout << ", " <<     local_translation.operator()(i,0);
	cout << "\n\ninv_local_translation =";	for (int i=0; i< 3; i++) cout << ", " << inv_local_translation.operator()(i,0);
	cout << "\n\nlocal_inv_pose ="; 		for (int i=0; i<16; i++) cout << ", " <<        local_inv_pose.operator()(i/4,i%4);
	return local_inv_pose;
																																			/*
																																			*  Inverse of a transformation matrix:
																																			*  http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0053
																																			*
																																			*   {     |   }-1       {       |        }
																																			*   {  R  | t }     =   {  R^T  |-R^T .t }
																																			*   {_____|___}         {_______|________}
																																			*   {0 0 0| 1 }         {0  0  0|    1   }
																																			*
																																			*/
}

void Dynamic_slam::getFrameData(){  // can load use separate CPU thread(s) ?
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num <<flush;
	R.copyTo(old_R);																														// get ground truth frame to frame pose transform
	T.copyTo(old_T);
	old_pose_GT 		= pose_GT;
	inv_old_pose_GT 	= inv_pose_GT;																										// NB Confirmed this copies the data  NOT just the pointer.
	old_K_GT			= K_GT;
	inv_old_K_GT		= inv_K_GT;
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n Dynamic_slam::getFrameData_chk 0.5, data file = " << txt[runcl.dataset_frame_num].c_str() << "\t" << flush;
																																			}
	std::string str = txt[runcl.dataset_frame_num].c_str();																					// grab .txt file from array of files (e.g. "scene_00_0000.txt")
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
	pose_GT 		= getPose(R, T);
	inv_pose_GT 	= getInvPose(pose_GT);
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::getFrameData_chk 0.1.2"<<flush;
																																				cout << "\npose_GT = (";
																																				for (int i=0; i<4; i++){
																																					for (int j=0; j<4; j++){
																																						cout << ", " << pose_GT.operator()(i,j);
																																					}cout << "\n     ";
																																				}cout << ")\n"<<flush;
																																				
																																				cout << "\ninv_pose_GT = (";
																																				for (int i=0; i<4; i++){
																																					for (int j=0; j<4; j++){
																																						cout << ", " << inv_pose_GT.operator()(i,j);
																																					}cout << "\n     ";
																																				}cout << ")\n"<<flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.2"<<flush;
	K_GT.zeros();
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			K_GT.operator()(i,j) = cameraMatrix.at<float>(i,j);
		}
	}K_GT.operator()(3,3) = 1;
	K = K_GT;
	generate_invK();
	inv_K_GT = inv_K; 																														// TODO change this when we autocalibrate K.
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.4"<<flush; // K2K
	K2K_GT = old_K_GT * old_pose_GT * inv_pose_GT * inv_K_GT;;
	//   wrong way around :  old_K_GT * old_pose_GT * inv_pose_GT * inv_K_GT;																				// TODO  Issue, not valid for first frame, pose  should be identty, Also what would estimate SE3 do ?
	
	pose2pose_GT = old_pose_GT * inv_pose_GT;
	/*
	if (runcl.costvol_frame_num <= 0 ) {																									// Transfered initialization to    dynamic_slam.initialize_keyframe_from_GT() or Dynamic_slam::initialize_keyframe_from_tracking()
		keyframe_pose_GT 		= pose_GT; 
		keyframe_inv_pose_GT 	= getInvPose(keyframe_pose_GT);
		keyframe_inv_K_GT		= generate_invK_(K_GT);				//  inv_old_K_GT;
	}
	*/
	keyframe_pose2pose_GT = pose_GT * keyframe_inv_pose_GT;
	keyframe_K2K_GT = K_GT * pose_GT * keyframe_inv_pose_GT * keyframe_inv_K_GT;
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::getFrameData_chk 0.1.2"<<flush;
																																				cout << "\truncl.costvol_frame_num = " << runcl.costvol_frame_num << flush;
																																				
																																				cout << "\nK_GT = (";
																																				for (int i=0; i<4; i++){
																																					for (int j=0; j<4; j++){
																																						cout << ", " << K_GT.operator()(i,j);
																																					}cout << "\n     ";
																																				}cout << ")\n"<<flush;
																																				
																																				cout << "\npose_GT = (";
																																				for (int i=0; i<4; i++){
																																					for (int j=0; j<4; j++){
																																						cout << ", " << pose_GT.operator()(i,j);
																																					}cout << "\n     ";
																																				}cout << ")\n"<<flush;
																																				
																																				cout << "\nkeyframe_pose_GT = (";
																																				for (int i=0; i<4; i++){
																																					for (int j=0; j<4; j++){
																																						cout << ", " << keyframe_pose_GT.operator()(i,j);
																																					}cout << "\n     ";
																																				}cout << ")\n"<<flush;
																																				
																																				cout << "\nkeyframe_inv_pose_GT = (";
																																				for (int i=0; i<4; i++){
																																					for (int j=0; j<4; j++){
																																						cout << ", " << keyframe_inv_pose_GT.operator()(i,j);
																																					}cout << "\n     ";
																																				}cout << ")\n"<<flush;
																																				
																																				cout << "\nkeyframe_inv_K_GT = (";
																																				for (int i=0; i<4; i++){
																																					for (int j=0; j<4; j++){
																																						cout << ", " << keyframe_inv_K_GT.operator()(i,j);
																																					}cout << "\n     ";
																																				}cout << ")\n"<<flush;
																																				
																																				cout << "\nkeyframe_K2K_GT = (";
																																				for (int i=0; i<4; i++){
																																					for (int j=0; j<4; j++){
																																						cout << ", " << keyframe_K2K_GT.operator()(i,j);
																																					}cout << "\n     ";
																																				}cout << ")\n"<<flush;
																																			}
	
	if (runcl.dataset_frame_num > 0 ) {  pose2pose_accumulated_GT = pose2pose_accumulated_GT * pose2pose_GT;	}							// Tracks pose tranform from first frame.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 1,"<<flush;
																																				
																																				cout << "\n\nold_K_GT = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< old_K_GT.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\n\nold_pose_GT = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< old_pose_GT.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\n\n##############################";
																																				
																																				cout << "\n\ninv_pose_GT = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< inv_pose_GT.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\n\ninv_K_GT = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< inv_K_GT.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\n\nK_GT = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< K_GT.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\n\n##############################";
																																				
																																				
																																				cout << "\n\nK2K_GT = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< K2K_GT.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\n\npose2pose_GT = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", "<< pose2pose_GT.operator()(i,j);
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
	int r = runcl.baseImage.rows;  //image.rows;
    int c = runcl.baseImage.cols;  //image.cols;
	depth_GT = loadDepthAhanda(depth[runcl.dataset_frame_num].string(), r,c,cameraMatrix);
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 3,"<<flush;}
	stringstream ss;
	stringstream png_ss;
	boost::filesystem::path folder_tiff = runcl.paths.at("depth_GT");
	string type_string = runcl.checkCVtype(depth_GT.type() );
	ss << "/" << folder_tiff.filename().string() << "_original_" << runcl.dataset_frame_num <<"type_"<<type_string;
	png_ss << "/" << folder_tiff.filename().string() << "_original_" << runcl.dataset_frame_num;
	boost::filesystem::path folder_png = folder_tiff;
	folder_tiff += ss.str();
	folder_tiff += ".tiff";
	folder_png  += "/png/";

	folder_png  += png_ss.str();
	folder_png  += ".png";
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 4,"<<flush;
																																				cout << "\n runcl.frame_num = " << runcl.dataset_frame_num << ",  depth[runcl.frame_num].string() = " << depth[runcl.dataset_frame_num].string() << flush;
																																				cout << "\n depth_GT.size() = " << depth_GT.size() << ",  depth_GT.type() = "<< type_string << ",  depth_GT.empty() = " <<  depth_GT.empty()   << flush;
																																				cout << "\n " << folder_png.string() << flush; 
																																			}
	cv::imwrite(folder_png.string(), depth_GT );
	cv::imwrite(folder_tiff.string(), depth_GT );
	
	//runcl.loadFrameData(depth_GT, K2K, pose2pose);
	runcl.load_GT_depth(depth_GT, invert_GT_depth);																							// loads to depth_mem_GT buffer.
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData finished,#################################################################"<<flush;
}

void Dynamic_slam::use_GT_pose(){
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::use_GT_pose_chk_0,"<<flush;	
	old_K		= old_K_GT;
	inv_K		= inv_K_GT;
	old_pose	= old_pose_GT;
	inv_pose	= inv_pose_GT;
	
	pose 		= pose_GT;
	inv_pose	= inv_pose_GT;
	K2K 		= keyframe_K2K_GT;
	pose2pose 	= keyframe_pose2pose_GT;
	
	for (int i=0; i<16; i++){ runcl.fp32_k2k[i] = K2K.operator()(i/4, i%4);}  
																																			if(verbosity>local_verbosity_threshold){ 
																																			
																																				cout << "\n\nK = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\n\ninv_K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< inv_K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\n\npose2pose = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< pose2pose.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\n\nruncl.fp32_k2k[i] = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< runcl.fp32_k2k[i*4 + j];
																																					}cout<<")";
																																				}cout<<flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::use_GT_pose finished,"<<flush;	
}

cv::Matx44f Dynamic_slam::generate_invK_(cv::Matx44f K_){
	int local_verbosity_threshold = 0;
	cv::Matx44f inv_K_;
	
	float fx   =  K_.operator()(0,0);
	float fy   =  K_.operator()(1,1);
	float skew =  K_.operator()(0,1);
	float cx   =  K_.operator()(0,2);
	float cy   =  K_.operator()(1,2);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nDynamic_slam::generate_invK_chk 1\n";
																																				cout<<"\nfx="<<fx <<"\nfy="<<fy <<"\nskew="<<skew <<"\ncx="<<cx <<"\ncy= "<<cy;
																																				cout << flush;
																																			}
	///////////////////////////////////////////////////////////////////// Inverse camera intrinsic matrix, see:
	// https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/#:~:text=Inverse,lines%20from%20the%20camera%20center.
	inv_K_ = inv_K_.zeros();
	inv_K_.operator()(0,0)  = 1.0/fx;  																										if(verbosity>local_verbosity_threshold) cout<<"\n1.0/fx="<<1.0/fx;
	inv_K_.operator()(1,1)  = 1.0/fy;  																										if(verbosity>local_verbosity_threshold) cout<<"\n1.0/fy="<<1.0/fy;
	inv_K_.operator()(2,2)  = 1.0;
	inv_K_.operator()(3,3)  = 1.0;

	inv_K_.operator()(0,1)  = -skew/(fx*fy);
	inv_K_.operator()(0,2)  = (cy*skew - cx*fy)/(fx*fy);
	inv_K_.operator()(1,2)  = -cy/fy;
																																			if(verbosity>local_verbosity_threshold) {
																																				cv::Matx44f test_K_ = inv_K_ * K_;
																																				cout<<"\n\ntest_camera_intrinsic_matrix inversion\n";	// Verify inv_K:
																																				for(int i=0; i<4; i++){
																																					for(int j=0; j<4; j++){
																																						cout<<"\t"<< std::setw(5)<<test_K_.operator()(i,j);
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

																																				cout<<"\n\nK_\n";
																																				for(int i=0; i<4; i++){
																																					for(int j=0; j<4; j++){
																																						cout<<"\t"<< std::setw(5)<<K_.operator()(i,j);
																																					}cout<<"\n";
																																				}cout<<"\n";

																																				cout<<"\n\ninv_K_\n";
																																				for(int i=0; i<4; i++){
																																					for(int j=0; j<4; j++){
																																						cout<<"\t"<< std::setw(5)<<inv_K_.operator()(i,j);
																																					}cout<<"\n";
																																				}cout<<"\n";
																																			}
	return inv_K_;
}


void Dynamic_slam::generate_invK(){ 																										// TODO hack this to work here 
	int local_verbosity_threshold = 1;
	
	float fx   =  K.operator()(0,0);
	float fy   =  K.operator()(1,1);
	float skew =  K.operator()(0,1);
	float cx   =  K.operator()(0,2);
	float cy   =  K.operator()(1,2);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nDynamic_slam::generate_invK_chk 1\n";
																																				cout<<"\nfx="<<fx <<"\nfy="<<fy <<"\nskew="<<skew <<"\ncx="<<cx <<"\ncy= "<<cy;
																																				cout << flush;
																																			}
	///////////////////////////////////////////////////////////////////// Inverse camera intrinsic matrix, see:
	// https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/#:~:text=Inverse,lines%20from%20the%20camera%20center.
	inv_K = inv_K.zeros();
	inv_K.operator()(0,0)  = 1.0/fx;  																										if(verbosity>local_verbosity_threshold) cout<<"\n1.0/fx="<<1.0/fx;
	inv_K.operator()(1,1)  = 1.0/fy;  																										if(verbosity>local_verbosity_threshold) cout<<"\n1.0/fy="<<1.0/fy;
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

void Dynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] ) {																				// Generates a set of 6 k2k to be used to compute the SE3 maps for the current camera intrinsic matrix.
	int local_verbosity_threshold = -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] ) chk_0" << endl << flush;
	// SE3 
	//const float res			= ( obj["cameraMatrix"][2].asFloat() + obj["cameraMatrix"][5].asFloat() ) /2.0;
	const float f			= ( obj["cameraMatrix"][0].asFloat() + obj["cameraMatrix"][4].asFloat() ) /2.0;									// focal length in pixels.
	const float delta 	  	= obj["ST3_delta"].asFloat() * obj["min_depth"].asFloat()  / f ;												// ST3_delta * (Translation to cause 1 pixel of parallax at min_depth)  	//1.0;//0.01; //0.001;  //  * obj["min_depth"].asFloat()
	const float delta_theta = obj["SO3_delta_theta"].asFloat() / f;																			// SO3_delta_theta * (Rotation to cause 1 pixel of rotation flow) //0.01; //0.001;
	const float cos_theta   = cos(delta_theta);
	const float sin_theta   = sin(delta_theta);
																																			// Old :  Rotate 0.01 radians i.e 0.573  degrees.  Translate 0.001 'units' of distance
																																			if(verbosity>local_verbosity_threshold){ cout << "\nDynamic_slam::generate_SE3_k2k( ) chk_1,"<<endl << flush;
																																				cout << "  delta_theta = "<<delta_theta<< " radians," <<endl << flush;
																																				cout << "  delta = "<<delta<< " units distance," <<endl << flush;
																																				cout << "  f = "<<f<<endl << flush;  // "  res = "<<res<< endl<<

																																				cout << "  obj[\"cameraMatrix\"][2] =  "      <<obj["cameraMatrix"][2].asFloat()       <<endl << flush;
																																				cout << "  obj[\"cameraMatrix\"][5] =  "      <<obj["cameraMatrix"][5].asFloat()       <<endl << flush;
																																				cout << "  obj[\"cameraMatrix\"][0] =  "      <<obj["cameraMatrix"][0].asFloat()       <<endl << flush;
																																				cout << "  obj[\"cameraMatrix\"][4] =  "      <<obj["cameraMatrix"][4].asFloat()       <<endl << flush;

																																				cout << "  obj[\"ST3_delta\"] =  "            <<obj["ST3_delta"].asFloat()             <<endl << flush;
																																				cout << "  obj[\"min_depth\"] =  "            <<obj["min_depth"].asFloat()             <<endl << flush;
																																				cout << "  obj[\"SO3_delta_theta\"] =  "      <<obj["SO3_delta_theta"].asFloat()       <<endl << flush;
																																			}
	//Identity =				(1,			0,			0,			0,  			0,			1,			0,			0,  			0,			0,			1,			0,  			0,	0,	0,	1);
	transform[Rx] = cv::Matx44f(1,         0,          0,          0,\
								0,         cos_theta, -sin_theta,  0,\
								0,         sin_theta,  cos_theta,  0,\
								0,         0,          0,          1);
	
	transform[Ry] = cv::Matx44f(cos_theta,   0,         sin_theta,  0,\
								0,           1,         0,          0,\
								-sin_theta,  0,         cos_theta,  0,\
								0,           0,         0,          1);
	
	transform[Rz] = cv::Matx44f(cos_theta, -sin_theta,  0,          0,\
								sin_theta,  cos_theta,  0,          0,\
								0,           0,         1,          0,\
								0,           0,         0,          1);
	
	transform[Tx] = cv::Matx44f(1,0,0,delta, 	0,1,0,0,		0,0,1,0,		0,0,0,1);
	transform[Ty] = cv::Matx44f(1,0,0,0, 		0,1,0,delta,	0,0,1,0,		0,0,0,1);
	transform[Tz] = cv::Matx44f(1,0,0,0, 		0,1,0,0,		0,0,1,delta,	0,0,0,1);
	
	cv::Matx44f cam2cam[6];
	for (int i=0; i<6; i++) { 
		cam2cam[i] = K*transform[i]*inv_K; 
																																			if(verbosity>local_verbosity_threshold) { 
																																				cout << "\ntransform["<<i<<"]=";
																																				for (int j=0; j<16; j++) cout << ", "<<transform[i].operator()(j/4,j%4); 
																																				cout << flush;
																																			}
	}
	
	for (int i=0; i<6; i++) {
		cam2cam[i] = K * transform[i] *  inv_K;
																																			if(verbosity>local_verbosity_threshold) { 
																																				cout << "\ncam2cam["<<i<<"]="; 
																																				for (int j=0; j<16; j++) cout << ", "<<cam2cam[i].operator()(j/4,j%4);
																																				cout << flush;
																																			}
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
																																			}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n\nDynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] )   finished" << endl << flush;
																																			}
}

/*
void Dynamic_slam::estimateSO3(){
	int local_verbosity_threshold = -1;
	const uint DoF = 3;
	const uint matxDoF = 9;
	const uint channels = 4;
	const uint lst_chan = 3;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::estimateSO3()_chk 0.0" << flush;}
	float Rho_sq_result=FLT_MAX,   old_Rho_sq_result=FLT_MAX,    next_layer_Rho_sq_result=FLT_MAX;
	uint layer = 5;
	float factor = 0.0025;
	for (int iter = 0; iter<10; iter++){ 																									// TODO step down layers if fits well enough, and out if fits before iteration limit. Set iteration limit param in config.json file.
		if (iter%2==0 && layer>1) {
			layer--;
			old_Rho_sq_result = next_layer_Rho_sq_result;
		}																																	if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSO3()_chk 0.5: iter="<<iter<<",  layer="<<layer << flush;}
		float SO3_results[8][DoF][channels] = {{{0,0,0,0}}};																				// TODO find better way to fix max num mimpap layers than just [8].
		float Rho_sq_results[8][channels] = {{0,0,0,0}};
		runcl.estimateSO3(SO3_results, Rho_sq_results, iter, runcl.mm_start, runcl.mm_stop);
																																			if(verbosity>local_verbosity_threshold) {cout << "\n Dynamic_slam::estimateSO3()_chk 0.6: iter="<<iter<<",  layer="<<layer << flush;
																																				cout << endl;
																																				for (int i=0; i<=runcl.mm_num_reductions+1; i++){ 															// results / (num_valid_px * img_variance) 
																																					cout << "\nDynamic_slam::estimateSO3(), Layer "<<i<<" SO3_results/num_groups = (";
																																					for (int k=0; k<DoF; k++){
																																						cout << "\t(";
																																						for (int l=0; l<lst_chan; l++){
																																							cout << ", \t" << SO3_results[i][k][l] / ( SO3_results[i][k][lst_chan]  *  runcl.img_stats[IMG_VAR+l]  );	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}
																																						cout << ", \t" << SO3_results[i][k][lst_chan] << ")";
																																					}cout << ")";
																																				}
																																			}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << endl;
																																				for (int i=0; i<=runcl.mm_num_reductions+1; i++){ 															// results / (num_valid_px * img_variance) 
																																					cout << "\nDynamic_slam::estimateSO3(), Layer "<<i<<" mm_num_reductions = "<< runcl.mm_num_reductions <<",  Rho_sq_results/num_groups = (";
																																					if (Rho_sq_results[i][lst_chan] > 0){
																																						for (int l=0; l<lst_chan; l++){  cout << ", \t" << Rho_sq_results[i][l] / ( Rho_sq_results[i][lst_chan]  *  runcl.img_stats[IMG_VAR+l]  );	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}cout << ", \t" << Rho_sq_results[i][lst_chan] << ")";
																																					}else{
																																						for (int l=0; l<lst_chan; l++){  cout << ", \t" << 0.0f  ;	// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																						}cout << ", \t" << Rho_sq_results[i][lst_chan] << ")";
																																					}
																																				}
																																			}
		uint channel = 0; 																													// TODO combine Rho HSV channels 
		
		Rho_sq_result = Rho_sq_results[layer][channel] / ( Rho_sq_results[layer][lst_chan]  *  runcl.img_stats[IMG_VAR+channel] );
		if (Rho_sq_results[layer+1][lst_chan]==0) {cout << "Dynamic_slam::estimateSO3(): Rho_sq_results[layer+1][lst_chan]==0" << flush;  break;}
		if (layer >0) { next_layer_Rho_sq_result  = Rho_sq_results[layer+1][channel] / ( Rho_sq_results[layer+1][lst_chan]  *  runcl.img_stats[IMG_VAR+channel] );}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\niter="<<iter<<", \tlayer="<<layer<<", \told_Rho_sq_result="<<old_Rho_sq_result<<",  \tRho_sq_result="<<Rho_sq_result <<",  \tnext_layer_Rho_sq_result="<< next_layer_Rho_sq_result <<flush;
																																			} 
		if (iter>0 && Rho_sq_result > old_Rho_sq_result) {																					// If new sample is worse, reject it. Continue to next iter. ? try a smaller step e.g. half size ?
																																			if(verbosity>local_verbosity_threshold) {cout << " (iter>0 && Rho_sq_result > old_Rho_sq_result)" << flush;} 
			//continue;
		} 
		
		old_Rho_sq_result = Rho_sq_result;
		float SO3_incr[DoF]; for (int SO3=0; SO3<DoF; SO3++) {SO3_incr[SO3] = SO3_results[5][SO3][channel] / ( SO3_results[5][SO3][lst_chan]  *  runcl.img_stats[IMG_VAR+channel]  );}			// For initial example take layer , channel[0] for each SO3 DoF.
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSO3()_chk 1" << flush;
																																				cout << "\n\nSO3_incr[SO3] = "; 	for (int SO3=0; SO3<DoF;SO3++) cout << ", \t" << SO3_incr[SO3];
																																				
																																				cout << "\n\nold pose2pose = "; 	for (int SE3=0; SE3<16; SE3++) cout << ", \t" << pose2pose.operator()(SE3/4,SE3%4);
																																				cout << "\n\nold K2K = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", \t" << K2K.operator()(SE3/4,SE3%4);
																																			}
		if (layer==1) factor *= 0.75;
		Matx31f update; for (int SO3=0; SO3<DoF; SO3++) {update.operator()(SO3) = factor * SO3_results[layer][SO3][channel] / ( SO3_results[layer][SO3][lst_chan] * runcl.img_stats[IMG_VAR+channel] ); }  // For each SO3 DoF, compute update increment, using "channel".
		cv::Matx33f SO3Incr_matx 	= SO3_Matx33f(update);																					// NB lst_chan holds ...?
		SO3_pose2pose 				= SO3_pose2pose * SO3Incr_matx;
		
		for (int i=0; i<matxDoF; i++) pose2pose.operator()(i/DoF,i%DoF) = SO3_pose2pose.operator()(i/DoF,i%DoF);
		
		K2K = old_K * pose2pose * inv_K;
		
		//for (int i=0; i<9; i++){ runcl.fp32_so3_k2k[i] = SO3_pose2pose.operator()(i/DoF, i%DoF); }
		for (int i=0; i<16; i++){ runcl.fp32_k2k[i] = K2K.operator()(i/4, i%4); }
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSO3()_chk 3\n" << flush;
																																				cout << "\n\nupdate.operator()(SO3) = "; 	for (int SO3=0; SO3<DoF; 	 SO3++) cout << ", \t" << update.operator()(SO3);
																																				cout << "\n\nSO3Incr_matx = ";				for (int SO3=0; SO3<matxDoF; SO3++) cout << ", \t" << SO3Incr_matx.operator()(SO3/DoF,SO3%DoF);
																																				cout << "\n\nSO3_pose2pose = "; 			for (int SO3=0; SO3<matxDoF; SO3++) cout << ", \t" << SO3_pose2pose.operator()(SO3/DoF,SO3%DoF);
																																				
																																				cout << "\n\npose2pose = "; 				for (int SE3=0; SE3<16; SE3++) cout << ", \t" << pose2pose.operator()(SE3/4,SE3%4);
																																				
																																				cout << "\n\nold_K = "; 					for (int SE3=0; SE3<16; SE3++) cout << ", \t" << old_K.operator()(SE3/4,SE3%4);
																																				cout << "\n\ninv_K = "; 					for (int SE3=0; SE3<16; SE3++) cout << ", \t" << inv_K.operator()(SE3/4,SE3%4);
																																				
																																				cout << "\n\nruncl.fp32_k2k = ";			for (int SE3=0; SE3<16; SE3++) cout << ", \t" << runcl.fp32_k2k[SE3];
																																			}
	}
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSO3()_chk finished\n" << flush;}
}
*/

void Dynamic_slam::update_k2k(Matx61f update_){
	int local_verbosity_threshold = -3;
																																				if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 1" << flush;
																																					cout << "\npose2pose = ";
																																					for (int i=0; i<4; i++){
																																						cout << "\n(";
																																						for (int j=0; j<4; j++){
																																							cout << ", \t"<< pose2pose.operator()(i,j);
																																						}cout<<")";
																																					}cout<<flush;

																																					cout << "\npose2pose_GT = ";
																																					for (int i=0; i<4; i++){
																																						cout << "\n(";
																																						for (int j=0; j<4; j++){
																																							cout << ", \t"<< pose2pose_GT.operator()(i,j);
																																						}cout<<")";
																																					}cout<<flush;
																																				}
	cv::Matx44f SE3Incr_matx = LieToP_Matx(update_); 								// 						= SE3_Matx44f(update_);
	pose2pose 										= pose2pose *  SE3Incr_matx;
	K2K 											= old_K * pose2pose * inv_K;
	for (int i=0; i<16; i++){ runcl.fp32_k2k[i] 	= K2K.operator()(i/4, i%4);   }
																																				if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 1" << flush;
																																				cout <<"\nupdate_ = ";
																																				for (int i=0; i<6; i++){cout << ", \t\t" << update_.operator()(0,i);}
																																				cout << flush;
																																				
																																				cout << "\nSE3Incr_matx = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< SE3Incr_matx.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\nNew K2K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< K2K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\nNew pose2pose = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< pose2pose.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\nold_K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< old_K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\ninv_K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< inv_K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				/*
																																				cout << "\nupdate.operator()(SE3) = "; 	for (int SE3=0; SE3< 6; SE3++) cout << ", \t" << update_.operator()(SE3);
																																				cout << "\nSE3Incr_matx = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", \t" << SE3Incr_matx.operator()(SE3/4,SE3%4);
																																				cout << "\nnew pose2pose = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", \t" << pose2pose.operator()(SE3/4,SE3%4);
																																				cout << "\nnew K2K = "; 				for (int SE3=0; SE3<16; SE3++) cout << ", \t" << K2K.operator()(SE3/4,SE3%4);
																																				*/
																																			}
}
/*
void Dynamic_slam::halve_update_k2k(Matx61f update_){
	int local_verbosity_threshold = -2;
	cv::Matx44f SE3Incr_matx 						= SE3_Matx44f(update_);
	pose2pose 										= pose2pose *  SE3Incr_matx;
	K2K 											= old_K * pose2pose * inv_K;
	for (int i=0; i<16; i++){ runcl.fp32_k2k[i] 	= K2K.operator()(i/4, i%4);   }
																																				if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::halve_update_k2k()_chk 1" << flush;
																																				cout << "\npose2pose = ";
																																				for (int i=0; i<16; i++) cout << ", \t" << pose2pose.operator()(i/4,i%4)  ;
																																				cout << "\n"<<flush;

																																				cout <<"\nupdate_ = ";
																																				for (int i=0; i<6; i++){cout << ", \t\t" << update_.operator()(0,i);}
																																				cout << flush;

																																				cout << "\nSE3Incr_matx = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< SE3Incr_matx.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\nNew K2K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< K2K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\nNew pose2pose = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< pose2pose.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\nold_K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< old_K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\ninv_K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< inv_K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				/ *
																																				cout << "\nupdate.operator()(SE3) = "; 	for (int SE3=0; SE3< 6; SE3++) cout << ", \t" << update_.operator()(SE3);
																																				cout << "\nSE3Incr_matx = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", \t" << SE3Incr_matx.operator()(SE3/4,SE3%4);
																																				cout << "\nnew pose2pose = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", \t" << pose2pose.operator()(SE3/4,SE3%4);
																																				cout << "\nnew K2K = "; 				for (int SE3=0; SE3<16; SE3++) cout << ", \t" << K2K.operator()(SE3/4,SE3%4);
																																				* /
																																			}
}
*/

void Dynamic_slam::estimateSE3(){
	int local_verbosity_threshold = -2;
	Matx61f old_update = {0,0,0, 0,0,0}, update = {0,0,0, 0,0,0};																			// SE3 Lie Algebra holding the DoF of SE3.
	cv::Matx44f SE3Incr_matx;																												// SE3 transformation matrix.
	if (obj["sample_se3_incr"]==true){
		initialize_resultsMat();
	}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::estimateSE3()_chk 0 : \t SE_factor = "<<SE_factor<<",\t obj[\"SE_factor\"].asFloat() = "<<obj["SE_factor"].asFloat() << flush;}
																																			// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
	float Rho_sq_result=FLT_MAX*0.99,   old_Rho_sq_result=FLT_MAX*0.99 ,   next_layer_Rho_sq_result=FLT_MAX*0.99;
	uint layer 		= SE3_start_layer;
	float factor 	= obj["SE_factor"].asFloat(); //SE_factor;

	uint channel  	= 2; 																													// TODO combine Rho HSV channels

	cout <<  "\nDynamic_slam::estimateSE3(): (Rho_sq_result < SE3_Rho_sq_threshold[layer][channel])=("<<Rho_sq_result<<" < "<<SE3_Rho_sq_threshold[layer][channel]<<")  ";
	for (int i=0; i<5; i++){ cout << " ( ";   for (int j=0; j<3; j++) {	cout << ", ["<<i<<"]["<<j<<"]" << SE3_Rho_sq_threshold[i][j]; }		cout << " ) "; }
	cout << ",\t layer = "<<layer<< ",\t factor = "<<factor << endl << flush;

	for (int iter = 0; iter<SE_iter; iter++){ 																								// TODO step down layers if fits well enough, and out if fits before iteration limit. Set iteration limit param in config.json file.
																																			if(verbosity>local_verbosity_threshold) {cout << "\n Dynamic_slam::estimateSE3()_chk 1.0" << "\t  iter = " << iter <<
																																				",\t layer = "<<layer<< ",\t factor = "<<factor<<flush;  }
		//////////////////////////////////////
		/*
		new algorithm
		1st sample Rho_sq, until(new_Rho < old Rho)
			{ reduce the step size 	}
		2nd measure gradients
		3rd compute the next step
		*/
		if (iter%3==0 && iter>0) {layer --; factor *= 0.75;}
		float 	Rho_sq_results[8][tracking_num_colour_channels] 	= {{0}};
		const float count[4] = {(float)iter, (float)layer, factor,0};

		runcl.se3_rho_sq(Rho_sq_results, count, runcl.mm_start, runcl.mm_stop);
		Rho_sq_result = Rho_sq_results[layer][channel] / ( Rho_sq_results[layer][3]  *  runcl.img_stats[IMG_VAR+channel] );

																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\nDynamic_slam::estimateSE3()_chk 1.2" << flush;
																																				for (int i=0; i<=runcl.mm_num_reductions+1; i++){ 					// Rho_sq_results / (num_valid_px * img_variance)
																																					cout << "\nDynamic_slam::estimateSE3()_chk 1.3:, Layer "<<i<<" mm_num_reductions = "<< runcl.mm_num_reductions <<
																																					",  \t\tRho_sq_results/num_groups = (";
																																					if (Rho_sq_results[i][3] > 0){
																																						for (int l=0; l<3; l++){  cout << ", \t" << Rho_sq_results[i][l] / ( Rho_sq_results[i][3]  *  runcl.img_stats[IMG_VAR+l]  );
																																						}
																																						cout << ", " << Rho_sq_results[i][3]  << ")\t\tIMG_VAR = ";
																																						for (int l=0; l<3; l++) cout << ", "<<runcl.img_stats[IMG_VAR+l];
																																					}else{
																																						for (int l=0; l<3; l++){  cout << ", \t" << 0.0f  ;
																																						}cout << ", " << Rho_sq_results[i][3] << ")";
																																					}
																																				}
																																			}

		/*
		cv::Matx44f oldpose2pose									= pose2pose;
		int 	iter_ 												= iter;

		for( ; iter_<SE_iter; iter_++) {																									if(verbosity>local_verbosity_threshold) {cout 	<< "\nDynamic_slam::estimateSE3()_chk 1.1" << "\t  iter_ = " << iter_ <<
																																				",\t layer = "<<layer<< ",\t factor = "<<factor<<flush; }
			runcl.se3_rho_sq(Rho_sq_results, iter_, runcl.mm_start, runcl.mm_stop);
																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\nDynamic_slam::estimateSE3()_chk 1.2" << flush;
																																				for (int i=0; i<=runcl.mm_num_reductions+1; i++){ 					// Rho_sq_results / (num_valid_px * img_variance)
																																					cout << "\nDynamic_slam::estimateSE3()_chk 1.7:, Layer "<<i<<" mm_num_reductions = "<< runcl.mm_num_reductions <<
																																					",  \t\tRho_sq_results/num_groups = (";
																																					if (Rho_sq_results[i][3] > 0){
																																						for (int l=0; l<3; l++){  cout << ", \t" << Rho_sq_results[i][l] / ( Rho_sq_results[i][3]  *  runcl.img_stats[IMG_VAR+l]  );
																																						}cout << ", " << Rho_sq_results[i][3] << ")";
																																					}else{
																																						for (int l=0; l<3; l++){  cout << ", \t" << 0.0f  ;
																																						}cout << ", " << Rho_sq_results[i][3] << ")";
																																					}
																																				}
																																			}
			Rho_sq_result = Rho_sq_results[layer][channel] / ( Rho_sq_results[layer][3]  *  runcl.img_stats[IMG_VAR+channel] );					// Rho_sq_result[channel] / (num_valid_pixels * image_variance[channel] ) : measure of fit.
																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\nDynamic_slam::estimateSE3()_chk 1.3: "<<
																																				"  \titer="<<iter<<
																																				", \tlayer="<<layer<<
																																				", \told_Rho_sq_result="<<old_Rho_sq_result<<
																																				", \tRho_sq_result="<<Rho_sq_result <<
																																				", \t\tnext_layer_Rho_sq_result="<< next_layer_Rho_sq_result <<
																																				flush;
																																				cout 	<< "\t\tDynamic_slam::estimateSE3()_chk 1.3:  pose2pose = ";
																																				for (int i=0; i<16; i++) {if (i%4 == 0) cout<<"\t\t "; cout << pose2pose.operator()(i/4,i%4) << ", \t";}
																																				cout << endl << flush;
																																			}
			float threshold = SE3_Rho_sq_threshold[layer][channel];
																																			cout <<  "\nDynamic_slam::estimateSE3(): (Rho_sq_result <? SE3_Rho_sq_threshold["<<layer<<"]["<<channel<<
																																				"])=("<<Rho_sq_result<<
																																				" < "<<(SE3_Rho_sq_threshold[layer][channel])<<
																																				") iter = "<<iter<<", iter_ = "<<iter_<<",  threshold = "<< threshold << flush;
			if (Rho_sq_result < (SE3_Rho_sq_threshold[layer][channel] ) ) {
				layer--;
				old_Rho_sq_result = Rho_sq_results[layer][channel] / ( Rho_sq_results[layer][3]  *  runcl.img_stats[IMG_VAR+channel] );
				break;
			}
			if (Rho_sq_result < old_Rho_sq_result) {break;}// NB always breaks for 1st iter,												// reduce step size
			factor 				= factor * 0.5;
			update 				= update * 0.5;
			pose2pose 			= oldpose2pose;				// reset previous iteration
			update_k2k(update);
			if (layer > 1) layer--;

			runcl.se3_rho_sq(Rho_sq_results, iter_, runcl.mm_start, runcl.mm_stop);
			break;

		}
		*/

		//////////////////////////////////////
																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\nDynamic_slam::estimateSE3()_chk 1.5: iter="<<iter<<
																																				",  layer="<<layer<< ",\t factor = "<<factor<<",\t Rho_sq_result = "<<Rho_sq_result<< flush;}
		float SE3_results[8][6][tracking_num_colour_channels] = {{{0}}};
		//float Rho_sq_results[8][tracking_num_colour_channels] = {{0}};

		runcl.estimateSE3(SE3_results, Rho_sq_results, iter, runcl.mm_start, runcl.mm_stop);
																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\nDynamic_slam::estimateSE3()_chk 1.6.0:" << flush;
																																				cout << endl;
																																				for (int i=0; i<=runcl.mm_num_reductions+1; i++){ 							// SE3_results / (num_valid_px * img_variance)
																																					cout 									<< "\nDynamic_slam::estimateSE3()_chk 1.6.1:"<<
																																					", Layer "<<i<<" SE3_results/num_groups = (";
																																					for (int k=0; k<6; k++){
																																						cout << "\t(";
																																						for (int l=0; l<3; l++){ cout << ", \t" << SE3_results[i][k][l]  ; }
																																						cout << ", \t" << SE3_results[i][k][3] << ")";
																																					}cout << ")";
																																					cout << "\t\t IMG_VAR = ";
																																					for (int l=0; l<3; l++) cout << " ,\t " << runcl.img_stats[IMG_VAR+l] ;
																																					cout << endl << flush;
																																				}
																																				cout 										<< "\nDynamic_slam::estimateSE3()_chk 1.6.2"<<
																																				"  \titer="<<iter<<
																																				", \tlayer="<<layer<<
																																				", \tnext_layer_Rho_sq_result="<< next_layer_Rho_sq_result <<
																																				", \tSE3_results["<<layer<<"][SE3]["<<channel<<"]=(\t"<<
																																				SE3_results[layer][0][channel]<<",\t"<<
																																				SE3_results[layer][1][channel]<<",\t"<<
																																				SE3_results[layer][2][channel]<<",\t"<<
																																				SE3_results[layer][3][channel]<<",\t"<<
																																				SE3_results[layer][4][channel]<<",\t"<<
																																				SE3_results[layer][5][channel]<<",\t"<<
																																				"), \tfactor="<<factor<<
																																				flush;
																																			}
																																					cout << "\n#### update = ";
		for (int SE3=0; SE3<6; SE3++) {
																																					cout << ", ("<<factor<<" * "<<SE3_results[layer][SE3][channel]<<" / ( "<<SE3_results[layer][SE3][3]<<" * "<<runcl.img_stats[IMG_VAR+channel] ;
			update.operator()(SE3) = factor * SE3_results[layer][SE3][channel] / ( SE3_results[layer][SE3][3] * runcl.img_stats[IMG_VAR+channel] );
																																					cout << " ) ) = \t "<< update.operator()(SE3) ;
		}																																			cout << flush;

																																					cout << "\t * min_depth = * "<<obj["min_depth"].asFloat()<<" = ,(, ";
		for (int SE3=3; SE3<6; SE3++) {
			update.operator()(SE3) *= obj["min_depth"].asFloat();
																																					cout << update.operator()(SE3) << " , ";
		}																																			cout << ") "<<endl<<flush;

																																			cout << "\n#### Rho_sq_result = "<<Rho_sq_result<<"\t update =   ";
																																			for (int SE3=0; SE3<6; SE3++) { cout << update.operator()(SE3) << ", \t";}
		update_k2k( update );																												if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3()_chk 6: (iter>0 && Rho_sq_result > old_Rho_sq_result)" << flush;}
		old_update 				= update;
		old_Rho_sq_result 		= Rho_sq_result;


		if (obj["sample_se3_incr"]==true){
			// Extract sample images from Rho_map and SE3_incr_map - done in RunCL::se3_rho_sq(..) and RunCL::estimate_SE3(..)
			// Write data into image
			uint reduction 			= obj["sample_layer"].asUInt();
			int cols 				= runcl.MipMap[reduction*8 + MiM_READ_COLS];
			int row_offset 			= 2* runcl.mm_margin ;
			int col_offset 			= 2* runcl.mm_margin + iter * (runcl.mm_margin + cols  );
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3()_chk 6.1" << flush;}
			stringstream ss;
			ss << "Rho_sq_result = " << Rho_sq_result << "\t";
			for (int se3 = 0; se3<6; se3++) {
				for (int layer = 0; layer<obj["num_reductions"].asInt(); layer ++){
					ss << SE3_results[layer][se3][2] << "\t";
				}ss << "\t";
			}
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3()_chk 6.2: \t"<< ss.str() << endl << flush;}

			// Writing over the Image
			//Point org(row_offset, col_offset);
			//putText( /*InputOutputArray*/ runcl.resultsMat, /*const String &*"Text On Image"*/ ss.str() , /*cv::Point*/ org,  FONT_HERSHEY_PLAIN, /*double fontScale*/ 0.1,  /*Scalar color*/ Scalar(0, 0, 255), /*int thickness=1*/ 0.1, /*int lineType=LINE_8*/ LINE_AA /* , bool bottomLeftOrigin=false */);
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3()_chk 6.3" << flush;}
			/*
			// show inage
			cv::namedWindow( "Dynamic_slam::estimateSE3()_chk 6: writeToResultsMat" , 0 );
			cv::imshow("Dynamic_slam::estimateSE3()_chk 6: writeToResultsMat" , runcl.resultsMat);
			cv::waitKey(-1);
			destroyWindow( "Dynamic_slam::estimateSE3()_chk 6: writeToResultsMat"  );
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3()_chk 6.4" << flush;}
			*/
			// save image
			//runcl.SaveMat(resultsMat, CV_16FC4,  folder_tiff,  show,  max_range, "resultsMat", count);

		}





/*
		if (Rho_sq_result>old_Rho_sq_result) {																								// If new sample is worse, resample at half the previous increment half "factor".
			update = -0.5*old_update;																										if(verbosity>local_verbosity_threshold) {
																																							cout << "\n\n Dynamic_slam::estimateSE3()_chk 2.0: iter="<<iter<<", layer="<<layer
																																																<<"(Rho_sq_result > old_Rho_sq_result)  "<<  Rho_sq_result<<" > "<<old_Rho_sq_result << "  ,  "
																																																<<"update = -0.5*old_update = ("
																																																<< update.operator()(0)<<", "
																																																<< update.operator()(1)<<", "
																																																<< update.operator()(2)<<", "
																																																<< update.operator()(3)<<", "
																																																<< update.operator()(4)<<", "
																																																<< update.operator()(5)<<") "
																																																<< ", \t Rho_sq_result = " << Rho_sq_result
																																																<< flush;
																																			}			// Dynamic_slam::estimateSE3()_chk 2.1
			old_update = update;
			old_Rho_sq_result = Rho_sq_result;
			factor *=0.75f;
			continue;
		}

		if ((Rho_sq_result < SE3_Rho_sq_threshold[layer][channel]) && (layer>SE3_stop_layer)) {												// Layer increment.
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n\n Dynamic_slam::estimateSE3()_chk 2.1: ((Rho_sq_result < SE3_Rho_sq_threshold[layer][channel]) && (layer>SE3_stop_layer))\t  layer="
																																				<<layer<<", layer--"<< flush;}
				layer--;
				Rho_sq_result = Rho_sq_results[layer][channel] / ( Rho_sq_results[layer][3]  *  runcl.img_stats[IMG_VAR+channel] );			// Read the next layer's Rho_sq_result, until find a layer to sample again OR finish optimization
		}

		else {																																if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3()_chk 2.2:  normal tracking loop"
																																																<<",    Rho_sq_result = "<<Rho_sq_result<<",   old_Rho_sq_result = "<<old_Rho_sq_result<< flush;}
		}
			for (int SE3=0; SE3<6; SE3++) {
			update.operator()(SE3) = factor * SE3_results[layer][SE3][channel] / ( SE3_results[layer][SE3][3] * runcl.img_stats[IMG_VAR+channel] );
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nDynamic_slam::estimateSE3()_chk 2.3: iter="<<iter<<", layer="<<layer
																																				<<",  \t update.operator()(SE3) = factor * SE3_results["<<layer<<"]["<<SE3<<"]["<<channel
																																				<<"] / ( SE3_results[layer][SE3][3] * runcl.img_stats[IMG_VAR+channel] ) = ("
																																				<< factor << " * "<< SE3_results[layer][SE3][channel]
																																				<< " ) \t/ ( "<< SE3_results[layer][SE3][3] << " * " << runcl.img_stats[IMG_VAR+channel]
																																				<< " ) = \t"
																																				<< factor * SE3_results[layer][SE3][channel] / ( SE3_results[layer][SE3][3] * runcl.img_stats[IMG_VAR+channel] )
																																				<< ", \t Rho_sq_result = " << Rho_sq_result
																																				<< flush;
																																			}
		}
		for (int SE3=3; SE3<6; SE3++) { update.operator()(SE3) *= obj["min_depth"].asFloat(); }

		update_k2k( update );																												if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3()_chk 2.4: (iter>0 && Rho_sq_result > old_Rho_sq_result)" << flush;}
		old_update 				= update;
		old_Rho_sq_result 		= Rho_sq_result;
*/
		/*
		float SE3_incr[6];
		for (int SE3=0; SE3<6; SE3++) {SE3_incr[SE3] = SE3_results[5][SE3][channel] / ( SE3_results[5][SE3][3]  *  runcl.img_stats[IMG_VAR+channel]  );}							// For initial example take layer , channel[0] for each SE3 DoF.
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3()_chk 1" << flush;
																																				cout << "\n\nSE3_incr[SE3] = "; 	for (int SE3=0; SE3< 6; SE3++) cout << ", \t" << SE3_incr[SE3];
																																				cout << "\n\nold pose2pose = "; 	for (int SE3=0; SE3<16; SE3++) cout << ", \t" << pose2pose.operator()(SE3/4,SE3%4);
																																				cout << "\n\nold K2K = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", \t" << K2K.operator()(SE3/4,SE3%4);
																																			}
		for (int SE3=0; SE3<6; SE3++) {
			for (int chan=1; chan<2; chan++ ){
				update.operator()(SE3)  = factor * SE3_results[layer][SE3][chan] / ( SE3_results[layer][SE3][3] * runcl.img_stats[IMG_VAR+chan] );									// Sum the updates from the Saturation and Value channels.
			}
		}
		SE3Incr_matx 	= SE3_Matx44f(update);
		pose2pose 		= pose2pose * SE3Incr_matx;
		K2K 			= old_K * pose2pose * inv_K;
		
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3()_chk 2" << flush;
																																				cout << "\npose2pose = ";
																																				for (int i=0; i<16; i++) cout << ", \t" << pose2pose.operator()(i/4,i%4)  ;
																																				cout << "\n"<<flush;
																																				
																																				cout <<"\nupdate = ";
																																				for (int i=0; i<6; i++){cout << ", \t\t" << update.operator()(0,i);}
																																				cout << flush;
																																				
																																				cout << "\nSE3Incr_matx = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< SE3Incr_matx.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\nNew K2K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< K2K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\nNew pose2pose = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< pose2pose.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\nold_K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< old_K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\ninv_K = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< inv_K.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																			}
		
		for (int i=0; i<16; i++){ runcl.fp32_k2k[i] = K2K.operator()(i/4, i%4);   }															//if(verbosity>local_verbosity_threshold) cout << "\n\nK2K ("<<i%4 <<","<< i/4<<") = "<< runcl.fp32_k2k[i]; }
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3()_chk 3\n" << flush;
																																				cout << "\nupdate.operator()(SE3) = "; 	for (int SE3=0; SE3< 6; SE3++) cout << ", \t" << update.operator()(SE3);
																																				cout << "\nSE3Incr_matx = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", \t" << SE3Incr_matx.operator()(SE3/4,SE3%4);
																																				cout << "\nnew pose2pose = "; 			for (int SE3=0; SE3<16; SE3++) cout << ", \t" << pose2pose.operator()(SE3/4,SE3%4);
																																				cout << "\nnew K2K = "; 				for (int SE3=0; SE3<16; SE3++) cout << ", \t" << K2K.operator()(SE3/4,SE3%4);
																																			}
		*/
		// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
		// # Pass prediction to lower layers. Does it fit better ?
		// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
	}
	// show inage
		cv::namedWindow( "Dynamic_slam::estimateSE3()_chk 6: writeToResultsMat" , 0 );
		cv::imshow("Dynamic_slam::estimateSE3()_chk 6: writeToResultsMat" , runcl.resultsMat);
		cv::waitKey(-1);
		destroyWindow( "Dynamic_slam::estimateSE3()_chk 6: writeToResultsMat"  );
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3()_chk 7\n" << flush;
																																				cout << "\nruncl.frame_num = "<<runcl.dataset_frame_num;
																																				cout << "\npose2pose_accumulated = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< pose2pose_accumulated.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																				
																																				cout << "\nNew pose2pose = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< pose2pose.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																			}
	if (runcl.dataset_frame_num > 0 ) pose2pose_accumulated = pose2pose_accumulated * pose2pose;
	if (obj["sample_se3_incr"]==true){
		// save resultsMat to file;

	}

																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::estimateSE3()_chk 8  Finished\n" << flush;}
}


void Dynamic_slam::estimateSE3_LK(){
	int local_verbosity_threshold = -2;
	Matx61f old_update = {0,0,0, 0,0,0}, update = {0,0,0, 0,0,0};																			// SE3 Lie Algebra holding the DoF of SE3.
	cv::Matx44f SE3Incr_matx;																												// SE3 transformation matrix.
	if (obj["sample_se3_incr"]==true){
		initialize_resultsMat();
	}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::estimateSE3_LK()_chk 0 : \t SE_factor = "<<SE_factor<<
																																				",\t obj[\"SE_factor\"].asFloat() = "<<obj["SE_factor"].asFloat() <<
																																				",\t SE_iter = " << SE_iter <<
																																				flush;}
																																			// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
	float Rho_sq_result	=FLT_MAX*0.99,   old_Rho_sq_result=FLT_MAX*0.99 ,   next_layer_Rho_sq_result=FLT_MAX*0.99;
	uint  layer 						= SE3_start_layer;
	float factor 						= obj["SE_factor"].asFloat();					//0.04
	float factor_layer_multiplier 		= obj["SE_factor_layer_multiplier"].asFloat();  //0.75
	int   iter_per_layer 				= obj["SE_iter_per_layer"].asInt();				//1

	uint channel  		= 2; 																												// TODO combine Rho HSV channels

	cout <<  "\nDynamic_slam::estimateSE3_LK(): (Rho_sq_result < SE3_Rho_sq_threshold[layer][channel])=("<<Rho_sq_result<<" < "<<SE3_Rho_sq_threshold[layer][channel]<<")  ";
	for (int i=0; i<5; i++){ cout << " ( ";   for (int j=0; j<3; j++) {	cout << ", ["<<i<<"]["<<j<<"]" << SE3_Rho_sq_threshold[i][j]; }		cout << " ) "; }
	cout << ",\t layer = "<<layer<< ",\t factor = "<<factor << endl << flush;



	for (int iter = 0; iter<SE_iter; iter++){ 																								// TODO step down layers if fits well enough, and out if fits before iteration limit. Set iteration limit param in config.json file.
																																			if(verbosity>local_verbosity_threshold) {cout << "\n Dynamic_slam::estimateSE3_LK_LK()_chk 1.0" << "\t  iter = " << iter <<
																																				",\t layer = "<<layer<< ",\t factor = "<<factor<<flush;  }
		//////////////////////////////////////
		if (iter%iter_per_layer==0 && iter>0 ) {if (layer>0) layer --; factor *= factor_layer_multiplier;}

		float SE3_weights[8][6][tracking_num_colour_channels] = {{{0}}};
		float SE3_results[8][6][tracking_num_colour_channels] = {{{0}}};
		float Rho_sq_results[8][tracking_num_colour_channels] = {{0}};

		runcl.estimateSE3_LK(SE3_results, SE3_weights, Rho_sq_results, iter, runcl.mm_start, runcl.mm_stop);
																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\nDynamic_slam::estimateSE3_LK()_chk 1.6.0:" << flush;
																																				cout << endl;
																																				for (int i=runcl.mm_start; i<=runcl.mm_stop; i++){ 							// SE3_results / (num_valid_px * img_variance)
																																					cout 									<< "\n\nDynamic_slam::estimateSE3_LK()_chk 1.6.1:"<<
																																					", Layer "<<i<<" SE3_results = (";   //   /SE3_weights
																																					for (int k=0; k<6; k++){
																																						cout << "\n(";
																																						for (int l=0; l<3; l++){ cout << ", \t" << SE3_results[i][k][l]   ; }  //   / SE3_weights [i][k][l]
																																						cout << ", \t" << SE3_results[i][k][3] << ")";
																																					}cout << ")";
																																					cout << "\t\t IMG_VAR = ";
																																					for (int l=0; l<3; l++) cout << " ,\t " << runcl.img_stats[IMG_VAR+l] ;
																																					cout << endl << flush;
																																				}
																																				cout 										<< "\nDynamic_slam::estimateSE3_LK()_chk 1.6.2"<<
																																				"  \titer="<<iter<<
																																				", \tlayer="<<layer<<
																																				", \tnext_layer_Rho_sq_result="<< next_layer_Rho_sq_result <<
																																				", \tSE3_results["<<layer<<"][SE3]["<<channel<<"]=(\t"<< flush;
																																				/*
																																				cout << "\n Pointers:  &SE3_results="<<&SE3_results<<  flush;
																																				cout << ",  &SE3_results["<<layer<<"]="<<&SE3_results[layer]<< flush;
																																				cout << ",  &SE3_results[layer][0]="<<&SE3_results[layer][0]<< flush;
																																				cout << ",  &SE3_results[layer][0]["<<channel<<"]="<<&SE3_results[layer][0][channel]<< flush;
																																				*/
																																				if (isnormal(SE3_results[layer][0][channel])) cout << SE3_results[layer][0][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][1][channel])) cout << SE3_results[layer][1][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][2][channel])) cout << SE3_results[layer][2][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][3][channel])) cout << SE3_results[layer][3][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][4][channel])) cout << SE3_results[layer][4][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][5][channel])) cout << SE3_results[layer][5][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				cout << "), \tfactor="<<factor<<
																																				flush;
																																			}
																																					cout << "\n#### update = ";
		for (int SE3=0; SE3<6; SE3++) { //6
																																					cout << ", \nupdate se3 dof "<<SE3<<", layer "<<layer<<" = ("<< SE3_update_dof_weights[SE3]<<" * "<<SE3_update_layer_weights[layer]<<" * "<<factor<<" * "<<SE3_results[layer][SE3][channel]<<" / ( "<<SE3_weights[layer][SE3][channel]<<" * "<<runcl.img_stats[IMG_VAR+channel] ;
			if(SE3<6)update.operator()(SE3) = SE3_update_dof_weights[SE3] * SE3_update_layer_weights[layer] * factor * SE3_results[layer][SE3][channel] / (SE3_weights[layer][SE3][channel] * runcl.img_stats[IMG_VAR+channel] ) ;							// apply se3_dim weights and global factor.

																																					cout << " ) ) = \t "<< update.operator()(SE3) ;
		}																																			cout << flush;

		update_k2k( update );																												if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3_LK()_chk 6: (iter>0 && Rho_sq_result > old_Rho_sq_result)" << flush;}
		old_update 				= update;
		old_Rho_sq_result 		= Rho_sq_result;

		if (obj["sample_se3_incr"]==true){
																																			// Extract sample images from Rho_map and SE3_incr_map - done in RunCL::se3_rho_sq(..) and RunCL::estimate_SE3(..)
																																			// Write data into image
																																			//uint reduction 			= obj["sample_layer"].asUInt();
																																			//int cols 				= runcl.MipMap[reduction*8 + MiM_READ_COLS];
																																			//int row_offset 			= 2* runcl.mm_margin ;
																																			// int col_offset 			= 2* runcl.mm_margin + iter * (runcl.mm_margin + cols  );
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3_LK()_chk 6.1" << flush;}
																																			stringstream ss;
																																			ss << "Rho_sq_result = " << Rho_sq_result << "\nSE3_results[layer][se3][chan=2 'value'] :";
																																			for (int se3 = 0; se3<6; se3++) { ss<< "\nse3 dof = "<< se3 << " : ";
																																				for (int layer = 0; layer<obj["num_reductions"].asInt(); layer ++){
																																					ss << SE3_results[layer][se3][2] << "  \t";
																																				}ss << "\t";
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n Dynamic_slam::estimateSE3_LK()_chk 6.2: \t"<< ss.str() << endl << flush;}
		}
		// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
		// # Pass prediction to lower layers. Does it fit better ?
		// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
	}
	cv::namedWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , 0 );														// show runcl.resultsMat
	cv::imshow("Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , runcl.resultsMat);
	cv::waitKey(-1);
	destroyWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" );
	stringstream ss;																														// save runcl.resultsMat
	ss <<  runcl.paths.at("SE3_rho_map_mem").string() << "resultsMat_"<<runcl.dataset_frame_num<<".png";
	cv::imwrite( ss.str(), runcl.resultsMat );
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::estimateSE3_LK()_chk 7\n" << flush;
																																				cout << "\nruncl.frame_num = "<<runcl.dataset_frame_num;
																																				cout << "\npose2pose_accumulated = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< pose2pose_accumulated.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;

																																				cout << "\nNew pose2pose = ";
																																				for (int i=0; i<4; i++){
																																					cout << "\n(";
																																					for (int j=0; j<4; j++){
																																						cout << ", \t"<< pose2pose.operator()(i,j);
																																					}cout<<")";
																																				}cout<<flush;
																																			}
	if (runcl.dataset_frame_num > 0 ) pose2pose_accumulated = pose2pose_accumulated * pose2pose;
	if (obj["sample_se3_incr"]==true){
		// save resultsMat to file;
	}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::estimateSE3_LK()_chk 8  Finished\n" << flush;}
}



void Dynamic_slam::estimateCalibration(){
// # Get 1st & 2nd order gradients wrt calibration parameters. 
//


// # Take one dammped least squares step of calibration.
//

}


///////////////////////////////////////////////////////////////////////////////////////////////

void Dynamic_slam::initialize_keyframe_from_GT(){																							// GT depth map is for current GT pose.
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::initialize_keyframe_from_GT()_chk 0" << flush;}
	keyframe_pose_GT 		= pose_GT;
	keyframe_inv_pose_GT 	= getInvPose(keyframe_pose_GT);
	keyframe_inv_K_GT		= generate_invK_(K_GT);	
	
	
	keyframe_pose 		= pose_GT;
	keyframe_K			= K_GT;
	keyframe_inv_pose 	= inv_pose_GT;
	keyframe_inv_K		= inv_K_GT;
	
	keyframe_old_K		= old_K_GT;
	keyframe_old_pose	= old_pose_GT;
	
	keyframe_K2K 		= K2K_GT;						// TODO chk wrt when this is called and what values it would hold.
	keyframe_pose2pose 	= pose2pose_GT;
	
	runcl.initializeDepthCostVol( runcl.depth_mem_GT );
	initialize_new_keyframe();
}

void Dynamic_slam::initialize_keyframe_from_tracking(){																						// NB need to transform depth map from previous keyfrae to current pose.
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::initialize_keyframe_from_tracking()_chk 0" << flush;}
	keyframe_old_pose		= keyframe_pose;
	keyframe_old_K			= keyframe_K;
	//keyframe_old_inv_pose 	= keyframe_inv_pose;		// These variables don't exist
	//keyframe_old_inv_K		= keyframe_inv_K;
	
	keyframe_pose 		= pose;
	keyframe_K			= K;
	keyframe_inv_pose 	= inv_pose;
	keyframe_inv_K		= inv_K;
	
	//keyframe_K2K 		= K2K;								// Wrong source for these variables ?
	//keyframe_pose2pose 	= pose2pose;
	
	cv::Matx44f inv_pose2pose = getInvPose(keyframe_pose2pose);																				//cv::Matx44f Dynamic_slam::getInvPose(cv::Matx44f pose) 
	cv::Matx44f forward_keyframe2K  = K * inv_pose2pose * inv_old_K;
																																			if(verbosity>local_verbosity_threshold){ 
																																				cout << "\nK = ";
																																				for (int i=0; i<16; i++) cout << ", " << K.operator()(i/4,i%4)  ;
																																				cout << "\n"<<flush;
																																				
																																				cout << "\nkeyframe_pose2pose = ";
																																				for (int i=0; i<16; i++) cout << ", " << keyframe_pose2pose.operator()(i/4,i%4)  ;
																																				cout << "\n"<<flush;
																																				
																																				cout << "\ninv_pose2pose = ";
																																				for (int i=0; i<16; i++) cout << ", " << inv_pose2pose.operator()(i/4,i%4)  ;
																																				cout << "\n"<<flush;
																																				
																																				cout << "\ninv_old_K = ";
																																				for (int i=0; i<16; i++) cout << ", " << inv_old_K.operator()(i/4,i%4)  ;
																																				cout << "\n"<<flush;
																																				
																																				cout << "\nforward_keyframe2K = ";
																																				for (int i=0; i<16; i++) cout << ", " << forward_keyframe2K.operator()(i/4,i%4)  ;
																																				cout << "\n"<<flush;
																																			}
	runcl.transform_depthmap(forward_keyframe2K, runcl.keyframe_depth_mem );																// NB runcl.transform_depthmap(..) must be used _before_ initializing the new cost_volume, because it uses keyframe_basemem.
	runcl.initializeDepthCostVol( runcl.depth_mem );
	initialize_new_keyframe();
}

void Dynamic_slam::initialize_new_keyframe(){
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::initialize_new_keyframe()_chk 0,  runcl.dataset_frame_num = "<< runcl.dataset_frame_num << flush;}
	runcl.initialize_fp32_params();
	//runcl.QD_count 	= 0; // TODO NB these are reset in Dynamic_slam::nextFrame()
	//runcl.A_count 	= 0;
	//runcl.G_count 	= 0;
	
	//cacheGValues();			// TODO may not be needed here.
								// TODO   keyframe_K2K_GT, keyframe_K2K etc ? 
	runcl.keyFrameCount++;
	runcl.dataset_frame_num++;
}

void Dynamic_slam::updateDepthCostVol(){																							// Built forwards. Updates keframe only when needed.
	int local_verbosity_threshold = -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::buildDepthCostVol()_chk 0,  runcl.dataset_frame_num="<<runcl.dataset_frame_num << flush;}
// # Build depth cost vol on current image, using image array[6] in MipMap buffer, plus RelVelMap, 
// with current camera params & DepthMap if bootstrapping, otherwise with params for each frame.
// NB 2*(1+7) = 14 layers on MipMap DepthCostVol: for model & pyramid, ID cetntre layer plus 7 samples, i.e. centre +&- 3 layers.
	
	// ############ From pseudocode.cpp
	// 5) (i)data exchange between mapping and tracking ?
	//    (ii)NB choice of relative scale - how many frames?
	//    (iii)Boostrapping ? (a) initially assume spherical middle distance.
	//                        (b) initial values of other maps ?

	// 6) Need to rotate costvol forward to new frame & decay the old cost.
	//    Need to build cost vol around depth range estimate from higher image columnm layer - ie 32 layers in cost vol , not 256 layers
	//    therefore cost vol depth varies per pixel.
	
// Select naive depth map
// See CostVol::updateCost(..), RunCL::calcCostVol(..) &  __kernel void BuildCostVolume2
																																			// int count: Iteration of for loop in this function. Here used to count num imgages use in costvol.
	cv::Matx44f K2K_ =  keyframe_K2K_GT; 					//TODO K2K; 		// needs keyframe_K2K from keyframe. 						// camera-to-camera transform for this image to the keyframe of this cost vol.
	//bool image_ = runcl.frame_bool_idx; 																									// Index to correct img pyramid buffer on device.
	
	runcl.updateDepthCostVol( K2K_, runcl.costvol_frame_num, runcl.mm_start, runcl.mm_stop  ); 										// NB in DTAM_opencl : void RunCL::calcCostVol(float* k2k,  cv::Mat &image)
																																			// in  void Dynamic_slam::estimateSE3() above : runcl.estimateSE3(SE3_reults, Rho_sq_results, iter, 0, 8);
																																			// -> mipmap_call_kernel( se3_grad_kernel, m_queue, start, stop );
	
}

void Dynamic_slam::buildDepthCostVol_fast_peripheral(){																						// Higher levels only, built on current frame. 
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold){ cout<<"\nDynamic_slam::buildDepthCostVol_fast_peripheral_chk0, " << flush;}
	
	
	
}

/*void Dynamic_slam::computeSigmas(float epsilon, float theta, float L, float &sigma_d, float &sigma_q ){
		float mu	= 2.0*std::sqrt((1.0/theta)*epsilon) /L;
		sigma_d		= mu / (2.0/ theta);
		sigma_q		= mu / (2.0*epsilon);
}
*/

void Dynamic_slam::updateQD(){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateQD_chk0, epsilon="<<runcl.fp32_params[EPSILON]<<" theta="<<runcl.fp32_params[THETA]<<flush;
	runcl.computeSigmas(runcl.fp32_params[EPSILON], runcl.fp32_params[THETA], obj["L"].asFloat(), runcl.fp32_params[SIGMA_D], runcl.fp32_params[SIGMA_Q] );
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateQD_chk1, epsilon="<<runcl.fp32_params[EPSILON]<<" theta="<<runcl.fp32_params[THETA]\
																																								<<" sigma_q="<<runcl.fp32_params[SIGMA_Q]<<" sigma_d="<<runcl.fp32_params[SIGMA_D]<<flush;
	runcl.updateQD(runcl.fp32_params[EPSILON], runcl.fp32_params[THETA], runcl.fp32_params[SIGMA_Q], runcl.fp32_params[SIGMA_D], runcl.mm_start, runcl.mm_stop);	
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateQD_chk3, epsilon="<<runcl.fp32_params[EPSILON]<<" theta="<<runcl.fp32_params[THETA]\
																																								<<" sigma_q="<<runcl.fp32_params[SIGMA_Q]<<" sigma_d="<<runcl.fp32_params[SIGMA_D]<<flush;
}

void Dynamic_slam::cacheGValues()
{
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nDynamic_slam::cacheGValues()" <<flush;}
	runcl.updateG(runcl.G_count, runcl.mm_start, runcl.mm_stop);
}

bool Dynamic_slam::updateA(){
	int local_verbosity_threshold = 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateA "<<flush;
	if (theta < 0.001 && old_theta > 0.001){  cacheGValues(); old_theta=theta; }		// If theta falls below 0.001, then G must be recomputed.
	// bool doneOptimizing = (theta <= thetaMin);
	
	runcl.updateA( runcl.fp32_params[LAMBDA], runcl.fp32_params[THETA],  runcl.mm_start, runcl.mm_stop );
	
	runcl.measureDepthFit(runcl.mm_start, runcl.mm_stop);
	
	runcl.fp32_params[THETA] *= obj["thetaStep"].asFloat();
	
	//return doneOptimizing;
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateA finished"<<flush;
	return false;
}









/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

void Dynamic_slam::getResult(){
	
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


