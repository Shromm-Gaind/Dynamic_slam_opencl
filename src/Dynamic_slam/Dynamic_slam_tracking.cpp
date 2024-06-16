#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;


void Dynamic_slam::report_GT_pose_error(){ 																									// TODO this could be done in Max44f, without back and forth coversion.
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::report_GT_pose_error"];// -2;
	pose2pose_accumulated_error_algebra = LieSub(PToLie(pose2pose_accumulated), PToLie(pose2pose_accumulated_GT) ); // TODO wrong fornula	//pose2pose_accumulated_error_algebra 	= pose2pose_accumulated_algebra - pose2pose_accumulated_GT_algebra;
	keyframe_pose2pose_error_algebra 	= LieSub(PToLie(keyframe_pose2pose), PToLie(keyframe_pose2pose_GT)); 								//pose2pose_error_algebra 				= pose2pose_algebra - pose2pose_GT_algebra;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::report_GT_error_chk 0\n" << flush;
																																				PRINT_MATX44F(pose2pose_accumulated_GT,);
																																				PRINT_MATX44F(pose2pose_accumulated,);
																																				PRINT_MATX16F(pose2pose_accumulated_error_algebra,);
																																				//
																																				PRINT_MATX44F(keyframe_pose2pose_GT,);
																																				PRINT_MATX44F(keyframe_pose2pose,);
																																				PRINT_MATX16F(keyframe_pose2pose_GT_algebra,);
																																				PRINT_MATX16F(keyframe_pose2pose_algebra,);
																																				PRINT_MATX16F(keyframe_pose2pose_error_algebra,);
																																			}
}

void Dynamic_slam::display_frame_resluts(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::display_frame_resluts"];// 1;																										if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::display_frame_resluts_chk 0\n" << flush;}
	stringstream ss;
	ss << "_p2p_error_" ;				for (int i=0; i<6; i++) ss << "," << pose2pose_error_algebra.operator()(i);
	ss << "_cumulative_error_" ;		for (int i=0; i<6; i++) ss << "," << pose2pose_accumulated_error_algebra.operator()(i); // TODO wrong formula
	cout << "\nDynamic_slam::display_frame_resluts "<< ss.str() << flush;
	runcl.tracking_result( "Dynamic_slam::display_frame_resluts" );
}

void Dynamic_slam::artificial_pose_error(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::artificial_pose_error"];//-2;																										if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::artificial_pose_error() chk_0"<<flush; }
	Matx16f pose_step_algebra;
	for (int SE3=0; SE3<6; SE3++)  pose_step_algebra.operator()(0,SE3) = obj["Artif_pose_err_algebra"][SE3].asFloat();
	Matx44f poseStep 	= LieToP_Matx(pose_step_algebra);																					if(verbosity>local_verbosity_threshold){ PRINT_MATX44F(poseStep,);	PRINT_MATX16F(pose_step_algebra,);	PRINT_MATX16F(PToLie(keyframe_pose2pose),True);  }
	keyframe_pose2pose 	= keyframe_pose2pose * poseStep;																					if(verbosity>local_verbosity_threshold){ PRINT_MATX16F(PToLie(keyframe_pose2pose),Start); }
	K2K 				= old_K * keyframe_pose2pose * inv_K;																				if(verbosity>local_verbosity_threshold){ PRINT_MATX44F(K2K,New);	PRINT_FLOAT_16(runcl.fp32_k2keyframe,Old); }// Add error of one step in the 2nd SE3 DoF.
	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] = K2K.operator()(i/4, i%4);  }														if(verbosity>local_verbosity_threshold){ PRINT_FLOAT_16(runcl.fp32_k2keyframe,New); cout << "\nDynamic_slam::artificial_pose_error()_finish ##############################################" << flush;}
}

void Dynamic_slam::predictFrame(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::predictFrame"];/* -2;*/														if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num  << flush;
																																				PRINT_MATX44F(K2K,Old);
																																				PRINT_MATX44F(pose2pose,Old);
																																				cout << "\nruncl.dataset_frame_num  = " << runcl.dataset_frame_num;
																																				cout << "\nruncl.costvol_frame_num  = " << runcl.costvol_frame_num;
																																				PRINT_MATX44F(keyframe_inv_old_pose,Old);
																																				PRINT_MATX44F(pose2pose,Old);
																																				PRINT_MATX44F(old_pose2pose,Old);
																																				PRINT_MATX44F(keyframe_old_pose2pose,Old);
																																				PRINT_MATX44F(keyframe_pose2pose,Old);
																																				PRINT_MATX44F(keyframe_old_K2K,Old);
																																				PRINT_MATX44F(K2K,Old);
																																				PRINT_MATX44F(keyframe_K2K,Old);
																																			}
	keyframe_inv_old_pose 			= getInvPose(keyframe_pose2pose);
	pose2pose 						= keyframe_inv_old_pose * keyframe_pose2pose;
	if (runcl.costvol_frame_num>0)	{
		pose2pose 					= pose2pose * getInvPose(old_pose2pose) * pose2pose;													// Only use accel if there are enough previous frames.
	}
	old_pose2pose					= pose2pose;
	keyframe_old_pose2pose			= keyframe_pose2pose;
	keyframe_pose2pose 				= keyframe_pose2pose * pose2pose;
	keyframe_old_K2K 				= keyframe_K2K;
	keyframe_K2K 					= K * keyframe_pose2pose * inv_old_K;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 1\n" << flush;
																																				cout << "\nruncl.dataset_frame_num  = " << runcl.dataset_frame_num;
																																				cout << "\nruncl.costvol_frame_num  = " << runcl.costvol_frame_num;
																																				PRINT_MATX44F(keyframe_inv_old_pose,New);
																																				PRINT_MATX44F(pose2pose,New);
																																				PRINT_MATX44F(old_pose2pose,New);
																																				PRINT_MATX44F(keyframe_old_pose2pose,New);
																																				PRINT_MATX44F(keyframe_pose2pose,New);
																																				PRINT_MATX44F(keyframe_old_K2K,New);
																																				PRINT_MATX44F(K2K,New);
																																				PRINT_MATX44F(keyframe_K2K,New);
																																			}

	////////////////////////////////////////////////////////////////////////////
	// keyframe_old_K 					= keyframe_K;			//= K;
	// keyframe_inv_old_K				= keyframe_inv_K;		//= inv_K;
	// keyframe_old_pose				= keyframe_pose;		//= pose;
	// keyframe_inv_old_pose			= keyframe_inv_pose;	//= inv_pose;


	//keyframe_pose2pose		= pose2pose;
	/*pose2pose_algebra_2		= pose2pose_algebra_1;
	pose2pose_algebra_1		= PToLie(pose2pose);
	Matx16f 	p2p_alg 	= LieSub(pose2pose_algebra_1, pose2pose_algebra_2);

	if (runcl.costvol_frame_num==0){ pose2pose_algebra_0	= 								(runcl.dataset_frame_num > 2)*(p2p_alg );}		// Only use accel if there are enough previous frames.
	else{ 							 pose2pose_algebra_0	= LieAdd(pose2pose_algebra_1, 	(runcl.dataset_frame_num > 2)*(p2p_alg));}

	pose2pose 				= LieToP_Matx(pose2pose_algebra_0 );
	K2K 					= old_K * pose2pose * inv_K;
	keyframe_K2K 			= K * pose * keyframe_inv_pose * inv_old_K;   					// TODO fix this ?  Pose should reflect intertia, i.e. expect the same step again.

	for (int i=0; i<16; i++){ runcl.fp32_k2k[i] = K2K.operator()(i/4, i%4); }
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 1\n" << flush;
																																				cout << "\nruncl.dataset_frame_num  = " << runcl.dataset_frame_num;
																																				cout << "\nruncl.costvol_frame_num  = " << runcl.costvol_frame_num;
																																				PRINT_MATX16F(pose2pose_algebra_2,);
																																				PRINT_MATX16F(pose2pose_algebra_1,);
																																				PRINT_MATX16F(pose2pose_algebra_0,);
																																				PRINT_MATX44F(pose2pose,);
																																				PRINT_MATX44F(K2K,);
																																				PRINT_MATX44F(keyframe_K2K,);
																																			}
	*/// kernel update DepthMap with RelVelMap

	// kernel predict new frame
};


///


cv::Matx44f Dynamic_slam::generate_invK_(cv::Matx44f K_){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::generate_invK_"];// 0;
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
																																				PRINT_MATX44F(test_K_,test_camera_intrinsic_matrix inversion);
																																				//PRINT_MATX44F(pose,);
																																				//PRINT_MATX44F(inv_old_pose,);
																																				PRINT_MATX44F(K_,);
																																				PRINT_MATX44F(inv_K,);
																																			}
	return inv_K_;
}

void Dynamic_slam::generate_invK(){ inv_K = generate_invK_(K);  }


void Dynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] ) {																				// Generates a set of 6 k2k to be used to compute the SE3 maps for the current camera intrinsic matrix.
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::generate_SE3_k2k"];// -2;
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
																																				print_json_float_9(obj, "cameraMatrix");
																																				cout << "  delta_theta = "	<<delta_theta	<< " radians," 								<<endl << flush;
																																				cout << "  delta = "		<<delta			<< " units distance," 						<<endl << flush;
																																				cout << "  f = "			<<f				<< " pixels," 								<<endl << flush;
																																				cout << "  obj[\"ST3_delta\"] =  "            <<obj["ST3_delta"].asFloat()				<<endl << flush;
																																				cout << "  obj[\"min_depth\"] =  "            <<obj["min_depth"].asFloat()				<<endl << flush;
																																				cout << "  obj[\"SO3_delta_theta\"] =  "      <<obj["SO3_delta_theta"].asFloat()		<<endl << flush;
																																			}
	//Identity =				(1,			0,			0,			0,  			0,			1,			0,			0,  			0,			0,			1,			0,  			0,	0,	0,	1);
	cv::Matx44f transform[6];
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
	for (int i=0; i<6; i++) {  cam2cam[i] = K*transform[i]*inv_K; 																			if(verbosity>local_verbosity_threshold) PRINT_MATX44F(transform[i],);	}

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


void Dynamic_slam::update_k2k(Matx16f update_){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::update_k2k"];// -3;
																																			if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 1, compute idealSE3Incr_algebra :" << flush;
																																				PRINT_MATX44F(keyframe_pose2pose,);		PRINT_MATX44F(keyframe_pose2pose_GT,);
																																				Matx16f keyframe_pose2pose_alg_ 		= PToLie(pose2pose);												PRINT_MATX16F(keyframe_pose2pose_alg_	,update_k2k()_chk 1);
																																				Matx16f keyframe_pose2pose_GT_alg_ 		= PToLie(pose2pose_GT);												PRINT_MATX16F(keyframe_pose2pose_GT_alg_,update_k2k()_chk 1);
																																				Matx16f idealSE3Incr_alg_ 				= LieSub(keyframe_pose2pose_alg_, keyframe_pose2pose_GT_alg_);		PRINT_MATX16F(idealSE3Incr_alg_			,update_k2k()_chk 1);
																																				Matx44f idealSE3Incr_ 					= LieToP_Matx(idealSE3Incr_alg_);									PRINT_MATX44F(idealSE3Incr_				,update_k2k()_chk 1);

																																				Matx44f keyframe_pose2pose_inv 			= keyframe_pose2pose.inv(); 										PRINT_MATX44F(keyframe_pose2pose_inv	,update_k2k()_chk 1);
																																				Matx44f idealSE3Incr 					= keyframe_pose2pose_GT * keyframe_pose2pose_inv;					PRINT_MATX44F(idealSE3Incr				,update_k2k()_chk 1);
																																				Matx16f idealSE3Incr_algebra 			= PToLie(idealSE3Incr); 											PRINT_MATX16F(idealSE3Incr_algebra		,update_k2k()_chk 1);
																																			}
	cv::Matx44f SE3Incr_matx = LieToP_Matx(update_); 																						// 						= SE3_Matx44f(update_);
	keyframe_pose2pose 									= keyframe_pose2pose *  SE3Incr_matx;
	keyframe_K2K 										= old_K * keyframe_pose2pose * inv_K;
	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] 	= keyframe_K2K.operator()(i/4, i%4);   }
																																			if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 2, compute K2K :" << flush;
																																				PRINT_MATX16F(update_			,update_k2k()_chk 2);
																																				PRINT_MATX44F(SE3Incr_matx		,update_k2k()_chk 2);
																																				PRINT_MATX44F(K2K				,update_k2k()_chk 2);
																																				PRINT_MATX44F(keyframe_pose2pose,update_k2k()_chk 2);
																																				PRINT_MATX44F(old_K				,update_k2k()_chk 2);
																																				PRINT_MATX44F(inv_K				,update_k2k()_chk 2);

																																				cout << "\n####################################### finished Dynamic_slam::update_k2k(Matx16f update_)"<<flush;
																																			}
}

void Dynamic_slam::update_k2k_3(Matx16f update_, float k2k_3_16[3][16] ){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::update_k2k"];
	Matx44f keyframe_pose2pose_3[3];
	Matx44f k2k_3[3];

	for(int i=0; i<3; i++){
		cv::Matx44f SE3Incr_matx = LieToP_Matx(update_); 																						// 						= SE3_Matx44f(update_);
		keyframe_pose2pose_3[i] 								= keyframe_pose2pose *  SE3Incr_matx;
		k2k_3[i] 												= old_K * keyframe_pose2pose * inv_K;
		for (int j=0; j<16; j++) { k2k_3_16[i][j] 				= k2k_3[i].operator()(j/4, j%4);   }
	}
}



void Dynamic_slam::estimateSE3(){																											// new version with adaptive step and halting
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::estimateSE3"];// -1;
/* Adaptive step
 * 1) runcl.estSE3()
 * 2) compute three increment matrices
 * 3) runcl.se3_rho_sq()
 * 4) compute ideal increment, i.e. damped
 *
 * Stopping rule per layer
 * 5) check motion size of update, if (<1pixel for this layer) then move to next layer.
*/
	Matx16f old_update = {0,0,0, 0,0,0}, update = {0,0,0, 0,0,0};																			// SE3 Lie Algebra holding the DoF of SE3.
	cv::Matx44f SE3Incr_matx;																												// SE3 transformation matrix.
	if (obj["sample_se3_incr"]==true){
		initialize_resultsMat();
	}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n ###  Dynamic_slam::estimateSE3_LK()_chk 0 : \t SE_factor = "<<SE_factor<<
																																				",\t obj[\"SE_factor\"].asFloat() = "<<obj["SE_factor"].asFloat() <<
																																				",\t SE_iter = " << SE_iter <<
																																				flush;}
																																			// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
	float Rho_sq_result	=FLT_MAX*0.99,   old_Rho_sq_result=FLT_MAX*0.99 ,   next_layer_Rho_sq_result=FLT_MAX*0.99;
	uint  layer 						= SE3_start_layer;
	float factor 						= obj["SE_factor"].asFloat();					//0.04
	float factor_layer_multiplier 		= obj["SE_factor_layer_multiplier"].asFloat();  //0.75
	float factor_iter_multiplier 		= obj["SE_factor_iter_multiplier"].asFloat();	//0.9
	int   iter_per_layer 				= obj["SE_iter_per_layer"].asInt();				//1
	uint channel  						= 2;

																																				if(verbosity>local_verbosity_threshold) {
																																				cout <<  "\n### Dynamic_slam::estimateSE3_LK(): (Rho_sq_result < SE3_Rho_sq_threshold[layer][channel])=("<<Rho_sq_result<<
																																				" < "<<SE3_Rho_sq_threshold[layer][channel]<<")  ";
																																				for(int i=0;i<5;i++){cout<<" ( "; for(int j=0;j<3;j++) {cout<<", ["<<i<<"]["<<j<<"]"<<SE3_Rho_sq_threshold[i][j];}	cout << " ) "; }
																																				cout << ",\t layer = "<<layer<< ",\t factor = "<<factor << endl << flush;
																																			}
	for (int iter = 0; iter<SE_iter; iter++){ 																								// TODO step down layers if fits well enough, and out if fits before iteration limit. Set iteration limit param in config.json file.
																																			if(verbosity>local_verbosity_threshold) {cout << "\n###  Dynamic_slam::estimateSE3_LK_LK()_chk 1.0" << "\t  iter = " << iter <<
																																				",\t layer = "<<layer<< ",\t factor = "<<factor<<flush;
																																			}
		//////////////////////////////////////
		if (iter%iter_per_layer==0 && iter>0 ) {if (layer>0) layer --; factor *= factor_layer_multiplier;}

		float SE3_weights[8][6][tracking_num_colour_channels] = {{{0}}};
		float SE3_results[8][6][tracking_num_colour_channels] = {{{0}}};
		float Rho_sq_results[8][tracking_num_colour_channels] = {{0}};

		runcl.estimateSE3_LK(SE3_results, SE3_weights, Rho_sq_results, iter, layer, layer+1);//runcl.mm_start, runcl.mm_stop);
																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\n###  Dynamic_slam::estimateSE3_LK()_chk 1.6.0:" << flush;
																																				cout << endl;
																																				for (int i=runcl.mm_start; i<=runcl.mm_stop; i++){ 							// SE3_results / (num_valid_px * img_variance)
																																					cout 									<< "\n\n###  Dynamic_slam::estimateSE3_LK()_chk 1.6.1:"<<
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
																																				cout 										<< "\n###  Dynamic_slam::estimateSE3_LK()_chk 1.6.2"<<
																																				"  \titer="<<iter<<
																																				", \tlayer="<<layer<<
																																				", \tnext_layer_Rho_sq_result="<< next_layer_Rho_sq_result <<
																																				", \tSE3_results["<<layer<<"][SE3]["<<channel<<"]=(\t"<< flush;

																																				if (isnormal(SE3_results[layer][0][channel])) cout << SE3_results[layer][0][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][1][channel])) cout << SE3_results[layer][1][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][2][channel])) cout << SE3_results[layer][2][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][3][channel])) cout << SE3_results[layer][3][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][4][channel])) cout << SE3_results[layer][4][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][5][channel])) cout << SE3_results[layer][5][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				cout << "), \tfactor="<<factor<<
																																				flush;

																																				cout << "\n";
																																				cout << ",\tRho_sq_results["<<layer<<"]["<<channel<<"] = ";
																																				if( isfinite(Rho_sq_results[layer][channel]) ) cout << Rho_sq_results[layer][channel] ;
																																				cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {		cout << "\n#### update = "; }

		for (int SE3=0; SE3<6; SE3++) { //6
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << ", \nupdate se3 dof "<<SE3<<", layer "<<layer
																																				<<" = ("<< SE3_update_dof_weights[SE3]<<" * "<<SE3_update_layer_weights[layer]<<" * "<<factor<<" * "<<SE3_results[layer][SE3][channel]
																																				<<" / ( "<<SE3_weights[layer][SE3][channel]<<" * "<<runcl.img_stats[IMG_VAR+channel] ;
																																			}
			update.operator()(SE3) = SE3_update_dof_weights[SE3] * SE3_update_layer_weights[layer] * factor * SE3_results[layer][SE3][channel] / (SE3_weights[layer][SE3][channel] * runcl.img_stats[IMG_VAR+channel] ) ;							// apply se3_dim weights and global factor.

																																			if(verbosity>local_verbosity_threshold) {		cout << " ) ) = \t "<< update.operator()(SE3) << flush;}
		}
		for (int SE3=0; SE3<6; SE3++) {																										// Exit if tracking fails #############################################################################
			if ( isfinite( update.operator()(SE3) ) ) continue;
			else {
				cout << "\n\nTracking failed,  isfinite( update.operator()("<<SE3<<") ) = " <<  isfinite( update.operator()(SE3) ) << endl<<endl<<flush;
				exit(1);
			}
		}
		// update_k2k( update );																												if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_LK()_chk 6: (iter>0 && Rho_sq_result > old_Rho_sq_result)" << flush;}
		float k2k_3_16[3][16] = {{0}};
		update_k2k_3( update, k2k_3_16 );

		float count[4];
		count[0] = iter;
		count[1] = layer;
		count[2] = factor;
		count[3] = 0;
		runcl.se3_rho_sq(Rho_sq_results, count, layer, layer+1, k2k_3_16 );																	// get rho_sq for three sample steps k2k_3_16.
																																			// void RunCL::se3_rho_sq(float Rho_sq_results[8][4], const float count[4], uint start, uint stop,  float k2k_3_16_[3][16]  )









		old_update 				= update;
		old_Rho_sq_result 		= Rho_sq_result;
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_LK()_chk 6.1" << flush;
																																				stringstream ss;
																																				ss << "\tRho_sq_result = " << Rho_sq_result << "\nSE3_results[layer][se3][chan=2 'value'] :";
																																				for (int se3 = 0; se3<6; se3++) { ss<< "\nse3 dof = "<< se3 << " : ";
																																					for (int layer = 0; layer<obj["num_reductions"].asInt(); layer ++){
																																						ss << SE3_results[layer][se3][2] << "  \t";
																																					}ss << "\t";
																																				cout << ss.str() << endl << flush;}
																																			}
		factor *= factor_iter_multiplier;
		// # TODO maybe ...
		// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
		// # Pass prediction to lower layers. Does it fit better ?
		// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
	}
																																			if(obj["sample_se3_incr"].asBool()==true) { cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 6.3, display and save ResultsMat\n" << flush;
																																				if(obj["sample_se3_incr::display"].asBool()==true){
																																					cv::namedWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , 0 );														// show runcl.resultsMat
																																					cv::imshow("Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , runcl.resultsMat);
																																					cv::waitKey(-1);
																																					destroyWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" );
																																				}
																																				stringstream ss;																														// save runcl.resultsMat
																																				ss <<  runcl.paths.at("SE3_rho_map_mem").string() << "resultsMat_"<<runcl.dataset_frame_num<<".png";
																																				cv::imwrite( ss.str(), runcl.resultsMat );
																																			}
																																			if(verbosity>local_verbosity_threshold) { cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 7\n" << flush;
																																				cout << "\nruncl.frame_num = "<<runcl.dataset_frame_num;
																																				PRINT_MATX44F(pose2pose_accumulated,);
																																				PRINT_MATX44F(pose2pose,);
																																				PRINT_MATX44F(keyframe_pose2pose,);
																																			}
	if (runcl.dataset_frame_num > 0 ) pose2pose_accumulated = pose2pose_accumulated * pose2pose; // TODO wrong formula.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 8  Finished ####################################\n" << flush;}
}




void Dynamic_slam::estimateSE3_LK(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::estimateSE3_LK"];// -1;

	Matx16f old_update = {0,0,0, 0,0,0}, update = {0,0,0, 0,0,0};																			// SE3 Lie Algebra holding the DoF of SE3.
	cv::Matx44f SE3Incr_matx;																												// SE3 transformation matrix.
	if (obj["sample_se3_incr"]==true){
		initialize_resultsMat();
	}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n ###  Dynamic_slam::estimateSE3_LK()_chk 0 : \t SE_factor = "<<SE_factor<<
																																				",\t obj[\"SE_factor\"].asFloat() = "<<obj["SE_factor"].asFloat() <<
																																				",\t SE_iter = " << SE_iter <<
																																				flush;}
																																			// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
	float Rho_sq_result	=FLT_MAX*0.99,   old_Rho_sq_result=FLT_MAX*0.99 ,   next_layer_Rho_sq_result=FLT_MAX*0.99;
	uint  layer 						= SE3_start_layer;
	float factor 						= obj["SE_factor"].asFloat();					//0.04
	float factor_layer_multiplier 		= obj["SE_factor_layer_multiplier"].asFloat();  //0.75
	float factor_iter_multiplier 		= obj["SE_factor_iter_multiplier"].asFloat();	//0.9
	int   iter_per_layer 				= obj["SE_iter_per_layer"].asInt();				//1
	uint channel  						= 2; 																								// TODO combine Rho HSV channels
																																			if(verbosity>local_verbosity_threshold) {
																																				cout <<  "\n### Dynamic_slam::estimateSE3_LK(): (Rho_sq_result < SE3_Rho_sq_threshold[layer][channel])=("<<Rho_sq_result<<
																																				" < "<<SE3_Rho_sq_threshold[layer][channel]<<")  ";
																																				for(int i=0;i<5;i++){cout<<" ( "; for(int j=0;j<3;j++) {cout<<", ["<<i<<"]["<<j<<"]"<<SE3_Rho_sq_threshold[i][j];}	cout << " ) "; }
																																				cout << ",\t layer = "<<layer<< ",\t factor = "<<factor << endl << flush;
																																			}
	//if( runcl.costvol_frame_num  >0 ) runcl.update_tracking_depthmap(); // TODO later put this at the end of updateDepthCostVol(); OR just use amem for tracking, and therefore update amem in RunCL::transform_depthmap(..).

	for (int iter = 0; iter<SE_iter; iter++){ 																								// TODO step down layers if fits well enough, and out if fits before iteration limit. Set iteration limit param in config.json file.
																																			if(verbosity>local_verbosity_threshold) {cout << "\n###  Dynamic_slam::estimateSE3_LK_LK()_chk 1.0" << "\t  iter = " << iter <<
																																				",\t layer = "<<layer<< ",\t factor = "<<factor<<flush;
																																			}
		//////////////////////////////////////
		if (iter%iter_per_layer==0 && iter>0 ) {if (layer>0) layer --; factor *= factor_layer_multiplier;}

		float SE3_weights[8][6][tracking_num_colour_channels] = {{{0}}};
		float SE3_results[8][6][tracking_num_colour_channels] = {{{0}}};
		float Rho_sq_results[8][tracking_num_colour_channels] = {{0}};

		runcl.estimateSE3_LK(SE3_results, SE3_weights, Rho_sq_results, iter, layer, layer+1);//runcl.mm_start, runcl.mm_stop);
																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\n###  Dynamic_slam::estimateSE3_LK()_chk 1.6.0:" << flush;
																																				cout << endl;
																																				for (int i=runcl.mm_start; i<=runcl.mm_stop; i++){ 							// SE3_results / (num_valid_px * img_variance)
																																					cout 									<< "\n\n###  Dynamic_slam::estimateSE3_LK()_chk 1.6.1:"<<
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
																																				cout 										<< "\n###  Dynamic_slam::estimateSE3_LK()_chk 1.6.2"<<
																																				"  \titer="<<iter<<
																																				", \tlayer="<<layer<<
																																				", \tnext_layer_Rho_sq_result="<< next_layer_Rho_sq_result <<
																																				", \tSE3_results["<<layer<<"][SE3]["<<channel<<"]=(\t"<< flush;

																																				if (isnormal(SE3_results[layer][0][channel])) cout << SE3_results[layer][0][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][1][channel])) cout << SE3_results[layer][1][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][2][channel])) cout << SE3_results[layer][2][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][3][channel])) cout << SE3_results[layer][3][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][4][channel])) cout << SE3_results[layer][4][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][5][channel])) cout << SE3_results[layer][5][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				cout << "), \tfactor="<<factor<<
																																				flush;

																																				cout << "\n";
																																				cout << ",\tRho_sq_results["<<layer<<"]["<<channel<<"] = ";
																																				if( isfinite(Rho_sq_results[layer][channel]) ) cout << Rho_sq_results[layer][channel] ;
																																				cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {		cout << "\n#### update = "; }
		for (int SE3=0; SE3<6; SE3++) { //6
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << ", \nupdate se3 dof "<<SE3<<", layer "<<layer
																																				<<" = ("<< SE3_update_dof_weights[SE3]<<" * "<<SE3_update_layer_weights[layer]<<" * "<<factor<<" * "<<SE3_results[layer][SE3][channel]
																																				<<" / ( "<<SE3_weights[layer][SE3][channel]<<" * "<<runcl.img_stats[IMG_VAR+channel] ;
																																			}
			update.operator()(SE3) = SE3_update_dof_weights[SE3] * SE3_update_layer_weights[layer] * factor * SE3_results[layer][SE3][channel] / (SE3_weights[layer][SE3][channel] * runcl.img_stats[IMG_VAR+channel] ) ;							// apply se3_dim weights and global factor.

																																			if(verbosity>local_verbosity_threshold) {		cout << " ) ) = \t "<< update.operator()(SE3) << flush;}
		}
		for (int SE3=0; SE3<6; SE3++) {																										// Exit if tracking fails #############################################################################
			if ( isfinite( update.operator()(SE3) ) ) continue;
			else {
				cout << "\n\nTracking failed,  isfinite( update.operator()("<<SE3<<") ) = " <<  isfinite( update.operator()(SE3) ) << endl<<endl<<flush;
				exit(1);
			}
		}
		update_k2k( update );																												if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_LK()_chk 6: (iter>0 && Rho_sq_result > old_Rho_sq_result)" << flush;}
		old_update 				= update;
		old_Rho_sq_result 		= Rho_sq_result;
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_LK()_chk 6.1" << flush;
																																				stringstream ss;
																																				ss << "\tRho_sq_result = " << Rho_sq_result << "\nSE3_results[layer][se3][chan=2 'value'] :";
																																				for (int se3 = 0; se3<6; se3++) { ss<< "\nse3 dof = "<< se3 << " : ";
																																					for (int layer = 0; layer<obj["num_reductions"].asInt(); layer ++){
																																						ss << SE3_results[layer][se3][2] << "  \t";
																																					}ss << "\t";
																																				cout << ss.str() << endl << flush;}
																																			}
		factor *= factor_iter_multiplier;
		// # TODO maybe ...
		// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
		// # Pass prediction to lower layers. Does it fit better ?
		// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
	}
																																			if(obj["sample_se3_incr"].asBool()==true) { cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 6.3, display and save ResultsMat\n" << flush;
																																				if(obj["sample_se3_incr::display"].asBool()==true){
																																					cv::namedWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , 0 );														// show runcl.resultsMat
																																					cv::imshow("Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , runcl.resultsMat);
																																					cv::waitKey(-1);
																																					destroyWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" );
																																				}
																																				stringstream ss;																														// save runcl.resultsMat
																																				ss <<  runcl.paths.at("SE3_rho_map_mem").string() << "resultsMat_"<<runcl.dataset_frame_num<<".png";
																																				cv::imwrite( ss.str(), runcl.resultsMat );
																																			}
																																			if(verbosity>local_verbosity_threshold) { cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 7,   runcl.dataset_frame_num="<<runcl.dataset_frame_num<<"\n" << flush;
																																				cout << "\nruncl.frame_num = "<<runcl.dataset_frame_num;
																																				PRINT_MATX44F(pose2pose_accumulated,);
																																				PRINT_MATX44F(pose2pose,);
																																				PRINT_MATX44F(keyframe_pose2pose,);
																																				Matx16f keyframe_pose2pose_algebra = PToLie(keyframe_pose2pose);
																																				PRINT_MATX16F (keyframe_pose2pose_algebra   ,  );
																																			}
	if (runcl.dataset_frame_num > 0 ) pose2pose_accumulated = pose2pose_accumulated * pose2pose; // TODO wrong formula.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 8  Finished ####################################\n" << flush;}
}
