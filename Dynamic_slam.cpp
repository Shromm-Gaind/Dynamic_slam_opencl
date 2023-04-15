#include "Dynamic_slam.h"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;


Dynamic_slam::~Dynamic_slam()
{
};

Dynamic_slam::Dynamic_slam(
        Json::Value obj_
    ): runcl(obj_)
{
	obj = obj_;
	verbosity 	= obj["verbosity"].asInt();
	frame_num 	= obj["data_file_offset"].asUInt();
																														if(verbosity>0) cout << "\n Dynamic_slam::Dynamic_slam_chk 0\n" << flush;
	stringstream ss0;
	ss0 << obj["data_path"].asString() << obj["data_file"].asString();
	rootpath = ss0.str();
	root = rootpath;
	
	if ( exists(root)==false )		{ cout << "Data folder "<< ss0.str()  <<" does not exist.\n" <<flush; exit(0); }
	if ( is_directory(root)==false ){ cout << "Data folder "<< ss0.str()  <<" is not a folder.\n"<<flush; exit(0); }
	if ( empty(root)==true )		{ cout << "Data folder "<< ss0.str()  <<" is empty.\n"		 <<flush; exit(0); }
																														if(verbosity>0) cout << "\n Dynamic_slam::Dynamic_slam_chk 1\n" << flush;
	// get lists of files
	get_all(root, ".txt",   txt);                            // gathers all filepaths with each suffix, into c++ vectors.
	get_all(root, ".png",   png);
	get_all(root, ".depth", depth);
																														if(verbosity>0) cout << "\n Dynamic_slam::Dynamic_slam_chk 2\n" << flush;
    if (verbosity>0) cout << "\nDynamic_slam::Dynamic_slam(): "<< png.size()  <<" .png images found in data folder.\t"<<flush;
	// set image params
	runcl.baseImage 	= imread(png[frame_num].string());																// NB ref for dimensions and data type.
	runcl.allocatemem();
	if (verbosity>0) cout << "\nDynamic_slam::Dynamic_slam(): runcl.baseImage.size() = "<< runcl.baseImage.size()  <<" runcl.baseImage.type() = " << runcl.baseImage.type() << "\t"<< runcl.checkCVtype(runcl.baseImage.type()) <<flush;

};



int Dynamic_slam::nextFrame()
{																														if(verbosity>0) cout << "\n Dynamic_slam::nextFrame_chk 0\n" << flush;
	predictFrame();
	getFrame();
	getFrameData();
	estimateSO3();
	estimateSE3(); 				// own thread ? num iter ?
	estimateCalibration(); 		// own thread, one iter.
	
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
	frame_num++;
	return(0);					// NB option to return an error that stops the main loop.
};

void Dynamic_slam::predictFrame()
{
// generate initial prediction while data loads. OR do this at the end of the loop ?
	// new_pose = old pose + vel*timestep + accel*timestep²
	
	// kernel update DepthMap with RelVelMap
	
	// kernel predict new frame
};

void Dynamic_slam::getFrame()  // can load use separate CPU thread(s) ?
{																														if(verbosity>0) cout << "\n Dynamic_slam::getFrame_chk 0\n" << flush;
// # load next image to buffer NB load at position [log_2 index]
// See CostVol::updateCost(..) & RunCL::calcCostVol(..) 
	image = imread(png[frame_num].string());
	if (image.type()!= runcl.baseImage.type() || image.size()!=runcl.baseImage.size() ) {
		cout<< "\n\nError: Dynamic_slam::getFrame(), frame_num = " << frame_num << " : missmatched. runcl.baseImage.size()="<<runcl.baseImage.size()<<", image.size()="<<image.size()<<", runcl.baseImage.type()="<<runcl.baseImage.type()<<", image.type()="<<image.type()<<"\n\n"<<flush;
		exit(0);
	}
	runcl.loadFrame( image );
	runcl.cvt_color_space( );
	runcl.mipmap();
	runcl.img_gradients();

// # Get 1st & 2nd order image gradients of MipMap
// see CostVol::cacheGValues(), RunCL::cacheGValue2 & __kernel void CacheG3

}

void Dynamic_slam::getFrameData()  // can load use separate CPU thread(s) ?
{
	
}

void Dynamic_slam::estimateSO3()
{
// # Get 1st & 2nd order gradients of SO3 wrt predicted pose.
//


// # Predict 1st least squares step of SO3
//
	
}

void Dynamic_slam::estimateSE3()
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


