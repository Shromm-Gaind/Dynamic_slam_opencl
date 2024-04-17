#ifndef DTAM_UTILS_HPP
#define DTAM_UTILS_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h> // req for types e.g. CV_BGR2GRAY
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;

struct m33 {
	float data[9];
};

struct m34 {
	float data[12];
};

struct float3 {
	float data[3];
};
static Mat  makeGray(Mat image){
    if (image.channels()!=1) {
        cvtColor(image, image, CV_BGR2GRAY);
    }
    return image;
}


/*
///   ERROR my SE3 & SO3 functions are incorrect. Rather use originals below.
////  TODO replace all functions with Matx--f , if I use them, otherwise remove them and dependency on OpenCV other tha Matx-- itself.
static Matx31f SO3_Algebra(const Matx33f& SO3_Matx){
    Matx31f SO3_Algebra;
    SO3_Algebra.operator()(0,0) = SO3_Matx.operator()(1,2)  - SO3_Matx.operator()(2,1);
    SO3_Algebra.operator()(1,0) = SO3_Matx.operator()(2,0)  - SO3_Matx.operator()(0,2);
    SO3_Algebra.operator()(2,0) = SO3_Matx.operator()(0,1)  - SO3_Matx.operator()(1,0);
    
    return  SO3_Algebra;
}

static Matx33f SO3_Matx33f(const Matx31f& SO3_Algebra){
    Matx33f SO3_Matx;
    SO3_Matx.operator()(0,0 )   = 1.0f;
    SO3_Matx.operator()(1,1 )   = 1.0f;
    SO3_Matx.operator()(2,2 )   = 1.0f;
    
    SO3_Matx.operator()(1,2)    = SO3_Algebra.operator()(0,0)  ;
    SO3_Matx.operator()(2,0)    = SO3_Algebra.operator()(1,0)  ;
    SO3_Matx.operator()(0,1)    = SO3_Algebra.operator()(2,0)  ;
    
    SO3_Matx.operator()(2,1)    = - SO3_Algebra.operator()(0,0);
    SO3_Matx.operator()(0,2)    = - SO3_Algebra.operator()(1,0);
    SO3_Matx.operator()(1,0)    = - SO3_Algebra.operator()(2,0);
    
    return  SO3_Matx;
}

static Matx61f SE3_Algebra(const Matx44f& SE3_Matx){
    Matx61f SE3_Algebra;
    SE3_Algebra.operator()(0,0) = SE3_Matx.operator()(1,2)  - SE3_Matx.operator()(2,1);
    SE3_Algebra.operator()(1,0) = SE3_Matx.operator()(2,0)  - SE3_Matx.operator()(0,2);
    SE3_Algebra.operator()(2,0) = SE3_Matx.operator()(0,1)  - SE3_Matx.operator()(1,0);
    SE3_Algebra.operator()(3,0) = SE3_Matx.operator()(0,3)  ;
    SE3_Algebra.operator()(4,0) = SE3_Matx.operator()(1,3)  ;
    SE3_Algebra.operator()(5,0) = SE3_Matx.operator()(2,3)  ;

    return  SE3_Algebra;
}

static Matx44f SE3_Matx44f(const Matx61f& SE3_Algebra){
    Matx44f SE3_Matx;
    SE3_Matx.operator()(0,0 )   = 1.0f;
    SE3_Matx.operator()(1,1 )   = 1.0f;
    SE3_Matx.operator()(2,2 )   = 1.0f;
    SE3_Matx.operator()(3,3 )   = 1.0f;
    
    SE3_Matx.operator()(1,2)    = SE3_Algebra.operator()(0,0)  ;
    SE3_Matx.operator()(2,0)    = SE3_Algebra.operator()(1,0)  ;
    SE3_Matx.operator()(0,1)    = SE3_Algebra.operator()(2,0)  ;
    
    SE3_Matx.operator()(2,1)    = - SE3_Algebra.operator()(0,0);
    SE3_Matx.operator()(0,2)    = - SE3_Algebra.operator()(1,0);
    SE3_Matx.operator()(1,0)    = - SE3_Algebra.operator()(2,0);
    
    SE3_Matx.operator()(0,3)    = SE3_Algebra.operator()(3,0)  ;
    SE3_Matx.operator()(1,3)    = SE3_Algebra.operator()(4,0)  ;
    SE3_Matx.operator()(2,3)    = SE3_Algebra.operator()(5,0)  ;
    
    return  SE3_Matx;
}
*/
////

static Mat make4x4(const Mat& mat){
    
    if (mat.rows!=4||mat.cols!=4){
        Mat tmp=Mat::eye(4,4,mat.type());
        tmp(Range(0,mat.rows),Range(0,mat.cols))=mat*1.0;

        return tmp;
    }else{
        return mat;
    }
}

static Mat rodrigues(const Mat& p){
    
    Mat tmp;
    Rodrigues(p,tmp);
    return tmp;
}

static void LieToRT(InputArray Lie, OutputArray _R, OutputArray _T){
    Mat p = Lie.getMat();
    _R.create(3,3,CV_32FC1);
    Mat R = _R.getMat();
    _T.create(3,1,CV_32FC1);
    Mat T = _T.getMat();
    if(p.cols==1){
        p = p.t();
    }
        
    rodrigues(p.colRange(Range(0,3))).copyTo(R);
    Mat(p.colRange(Range(3,6)).t()).copyTo(T);
}

static void RTToLie(InputArray _R, InputArray _T, OutputArray Lie ){

    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Lie.create(1,6,T.type());
    
    Mat p = Lie.getMat(); 
    assert(p.size()==Size(6,1));
    p=p.reshape(1,6);
    if(T.rows==1){
        T = T.t();
    }
    
    rodrigues(R).copyTo(p.rowRange(Range(0,3)));
    T.copyTo(p.rowRange(Range(3,6)));
    assert(Lie.size()==Size(6,1));
}

static void RTToLie(Matx33f R, Matx13f T, Matx61f Lie ){
    Matx13f r(0,0,0);
    cv::Rodrigues(R, r);
    Lie.operator()(0) = r.operator()(0);
    Lie.operator()(1) = r.operator()(1);
    Lie.operator()(2) = r.operator()(2);

    Lie.operator()(3) = T.operator()(0);
    Lie.operator()(4) = T.operator()(1);
    Lie.operator()(5) = T.operator()(2);
}

static Mat RTToLie(InputArray _R, InputArray _T){
    Mat P;
    RTToLie(_R,_T,P);
    return P;
}

static void PToLie(InputArray _P, OutputArray Lie){
    Mat P = _P.getMat();
    assert(P.cols == P.rows && P.rows == 4);
    Mat R = P(Range(0,3),Range(0,3));
    Mat T = P(Range(0,3),Range(3,4));
    RTToLie(R,T,Lie);
    assert(Lie.size()==Size(6,1));
}

static void PToLie(Matx44f P, Matx61f Lie){
    Matx33f R;
    Matx13f T;
    for (int row=0; row<3;row++)for(int col=0; col<3; col++) R.operator()(row,col) = P.operator()(row,col);
    int col =3;                 for(int row=0; row<3;row++)  T.operator()(row,col) = P.operator()(row,col);
    RTToLie(R,T,Lie);
}

static void RTToP(InputArray _R, InputArray _T, OutputArray _P ){
    
    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Mat P = _P.getMat();
    hconcat(R,T,P);
    make4x4(P).copyTo(_P);
}
static Mat RTToP(InputArray _R, InputArray _T){
    
    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Mat P;
    hconcat(R,T,P);
    make4x4(P);
    return P;
}
static void LieToP(InputArray Lie, OutputArray _P){
    Mat p = Lie.getMat();
    _P.create(4,4,p.type());
    Mat P = _P.getMat();
    if(p.cols==1){
        p = p.t();
    } 
    
    Mat R=rodrigues(p.colRange(Range(0,3))); // makes rotation Mat from SO3 Lie vector.
    Mat T=p.colRange(Range(3,6)).t();
    hconcat(R,T,P);
    make4x4(P).copyTo(_P);
}

static void LieToP_Matx(Matx61f Lie, Matx44f P){
    Matx13f r;
    r.operator()(0) = Lie.operator()(0);
    r.operator()(1) = Lie.operator()(1);
    r.operator()(2) = Lie.operator()(2);
    Matx33f R;
    Rodrigues(r,R);                         // makes rotation Mat from SO3 Lie vector.
    for (int row=0; row<3;row++)for(int col=0; col<3; col++) P.operator()(row,col) = R.operator()(row,col);
    int col =3;                 for(int row=0; row<3;row++)  P.operator()(row,col) = Lie.operator()(col);
}

static Mat LieToP(InputArray Lie){
    Mat P;
    LieToP(Lie,P);
    return P;
}

static Mat LieSub(Mat A, Mat B){
    Mat Pa;
    Mat Pb;
    LieToP(A,Pa);
    LieToP(B,Pb);
    Mat out;
    assert(A.size()==Size(6,1) && B.size()==Size(6,1));
    PToLie(Pa*Pb.inv(),out);
    return out;
}

static Matx61f LieSub(Matx61f A, Matx61f B){
    Matx44f Pa;
    Matx44f Pb;
    LieToP_Matx(A,Pa);
    LieToP_Matx(B,Pb);
    Matx61f out;
    PToLie(Pa*Pb.inv(),out);
    return out;
}

static Mat LieAdd(Mat A, Mat B){
    Mat Pa;
    Mat Pb;
    
    LieToP(A,Pa);
    LieToP(B,Pb);
    Mat out;
    PToLie(Pa*Pb,out);
    return out;
}

static Matx61f LieAdd(Matx61f A, Matx61f B){
    Mat Pa;
    Mat Pb;

    LieToP(A,Pa);
    LieToP(B,Pb);
    Mat out;
    PToLie(Pa*Pb,out);
    return out;
}

template<class tp>
tp median_(const Mat& _M) {
    Mat M=_M.clone();
    int iSize=M.cols*M.rows;
    tp* dpSorted=(tp*)M.data;
    // Allocate an array of the same size and sort it.
    
    std::sort (dpSorted, dpSorted+iSize);

    // Middle or average of middle values in the sorted array.
    tp dMedian = 0.0;
    if ((iSize % 2) == 0) {
        dMedian = (dpSorted[iSize/2] + dpSorted[(iSize/2) - 1])/2.0;
    } else {
        dMedian = dpSorted[iSize/2];
    }
    return dMedian;
}

static double median(const Mat& M) { // NB only used for tp median_(const Mat& _M) above, which recasts to type tp.
    if(M.type()==CV_32FC1)
        return median_<float>(M);
    if(M.type()==CV_64FC1)
        return median_<double>(M);
    if(M.type()==CV_32SC1)
        return median_<int>(M);
    if(M.type()==CV_16UC1)
        return median_<unsigned int>(M);
    assert(!"Unsupported type");
}
#endif 
