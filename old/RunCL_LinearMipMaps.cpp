#include "RunCL.h"     // Shelve linearMipMap for now. It will not make kernels simpler or faster.

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

using namespace std;

/*
 Need to store 2D array of linear mipmap params

 When reading linearMipMap,
 1) download data
 2) copy data array into new image
    Use mipmap_params to locate data in new Mat.
 3) save png (&tiff)

 Issues
 1) data type
 2) num channels
 3) num maps in volume

 When writing CPU data to MipMap buffer
 1)

 When reading/writing linear Mipmap in kernels
 1) Minimize calculation
        read index = UID + offset
 2) Avoid smearing margins

 Make set of function aliases for each buffer.
 */

//void DownloadAndSave_linear_Mipmap(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, int type_mat, bool show );

void RunCL::DownloadAndSave_linearMipMap (
    cl_mem buffer,
    std::string count,
    //boost::filesystem::path folder_tiff,
    std::map< std::string, boost::filesystem::path > folder_tiff,   // used in DownloadAndSave_8Channel, when called by DownloadAndSave_8Channel_volume

    size_t      image_size_bytes,
    cv::Size    size_mat,

    int type_mat_out,           / *(data size and channels)* /
    int num_channels_out,       // channels per image to write to file
    int num_channels_in,        // channels in the buffer
    int maps_in_vol,            // layers of depth cost vol,  //uint vol_layers,

    int start_layer,            // of the mipmap
    int stop_layer,

    float max_range,            // used to scale the values in the png for visualization
    bool exception_tiff,        // generate tiff, when tiffs are globally turned off
    bool show                   // display the image(s).

    //uint offset =0,       // passed to ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0)
    //cv::Mat temp_mat,           // mat passed to SaveMat or SaveMat_1chan
    //std::string mat_name        //

){
    int type_mat_in = CV_MAKETYPE(CV_32F, num_channels_in);                                 // create temp_mat1
    cv::Mat tempMat1(size_mat, type_mat_in);
    float * data1 = (float*)tempMat1.data;
    ReadOutput(tempMat1.data, buffer, image_size_bytes /*,size_t offset=0*/);               // download buffer, to *uchar

    std::vector<cv::Mat>    tempMat2_vec;                                                   // create tempMat2_vec
    std::vector<float*>     data2;

    int num_Mats = num_channels_in/num_channels_out;
    for (int i=0; i<num_Mats; i++){
        tempMat2_vec.push_back(cv::Mat::zeros(size_mat, CV_32FC4));
        data2.push_back( (float*)tempMat2_vec[i].data );
    }

    for (int i=0; i<tempMat1.total(); i++){                                                 // write temp_mat1 to vector_temp_mat2
        int a = i * num_channels_in;
        int b = i * num_channels_out;
        for (int j=0; j<num_Mats; j++){
            a += j*num_channels_out;
            for (int k=0; k<num_channels_out; k++){
                data2[j][b+k] = data1[a+k];
            }
        }
    }

    std::vector<cv::Mat>    tempMat3_vec;                                                   // create tempMat3_vec
    std::vector<float*>     data3;
    cv::Size size_mipmap_mat(mm_height, mm_width );
///////////////////
    for (int i=0; i<num_Mats; i++){
        tempMat3_vec.push_back(cv::Mat::zeros(size_mipmap_mat, CV_32FC4));
        data3.push_back( (float*)tempMat3_vec[i].data );
    }

    for (int i=0; i<num_Mats; i++){
        for (int j=start_layer; j<stop_layer; j++){                                             // reconstruct mipmap
            cv:Size size_mmlayer_mat(   );
            cv::Mat tempMat4 = cv::Mat::zeros(size_mmlayer_mat, CV_32FC4);

            int offset_2    = ;  // Location of the mipmap layer in each Mat.
            int offset_3    = ;
            int rows        = ;
            int cols        = ;
            int margin      = ;

            for (int k=0; k<rows; k++){
                for (int l=0; l<cols; l++){
                    tempMat3_vec[i].at<Vec4f>(row,col)      = {data2[i].data[], data2[i].data[], data2[i].data[], data2[i].data[] };
                }
            }
        }
    }

    // save png(s)

    // save tiff(s)

    // display image
}


void RunCL::DownloadAndSave_... (){
    DownloadAndSave_linearMipMap (
    buffer              =,          /*cl_mem */
    count               =,          /*std::string*/
    folder_tiff         =,          /*boost::filesystem::path*/
    folder_tiff         =,          /*std::map< std::string, boost::filesystem::path >*/    // used in DownloadAndSave_8Channel, when called by DownloadAndSave_8Channel_volume

    image_size_bytes    =,          /*size_t*/
    size_mat            =,          /*cv::Size*/

    type_mat            =,          /*int*/                                                 // data size and channels
    num_channels_out    =,          /*int*/                                                 // channels per image to write to file
    num_channels_in     =,          /*int*/                                                 // channels in the buffer
    maps_in_vol         =,          /*int*/                                                 // layers of depth cost vol,  //uint vol_layers,

    start_layer         =,          /*int*/                                                 // of the mipmap
    stop_layer          =,          /*int*/

    //uint offset /*=0*/,                                                                   // passed to ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0)
    max_range           =,          /*float*/                                               // used to scale the values in the png for visualization

    exception_tiff      =,          /*bool*/                                                // generate tiff, when tiffs are globally turned off
    show                =,          /*bool*/                                                // display the image(s).

    temp_mat            =,          /*cv::Mat*/                                             // mat passed to SaveMat or SaveMat_1chan
    mat_name            =           /*std::string*/
    )
}



/* The old system, all are currently 32bit float, (could be 16bit in future, would require some brand specific code)
 *
    void DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );
    depthmap_GT, keyframe_depth_mem, key_frame_depth_map_src, lomem, himem, amem, dmem, qmem, gxmem
# 1Channel: 1 channel_out, 1 channel_in, 1 map.

	void DownloadAndSave_2Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	SE3_map_mem
# 2Channel: 3or4 channels_out,

	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1, uint offset=0, bool exception_tiff=false );
	basemem, imgmem, imgmem_blurred, gxmem, gymem, g1mem, keyframe_g1mem, dmem_disparity, buffer in DownloadAndSave_3Channel_volume,
# 3Channel: 4 channels_out (inc alpha), 4 channels_in, 1 map

	void DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	SE3_incr_map_mem, SE3_rho_map_mem   ## bug in .png output.
# 3Channel_volume: 2 channels_out (+ve & -ve), 3 channels_in,

	void DownloadAndSave_6Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0);
	buffer in DownloadAndSave_6ChannelVolume(..)

	void DownloadAndSave_6Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	SE3_grad_map_mem, keyframe_SE3_grad_map_mem

	void DownloadAndSave_HSV_grad(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0 );
	HSV_grad_mem, keyframe_imgmem,

	void SaveMat(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	mat_u & mat_v                          in DownloadAndSave_6Channel(..),
	Mat_H, mat_SV, mat_Sgrad, mat_Vgrad    in DownloadAndSave_HSV_grad(..)
# HSV_grad:         2 channels_out (per image),  8 channels_in, 1 map
# 6Channel_volume:  1 channels_out (per image),  6 channels_in, 1 map


	void DownloadAndSave_8Channel(cl_mem buffer, std::string count, std::map< std::string, boost::filesystem::path > folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range , uint offset );
	buffer in DownloadAndSave_8ChannelVolume(..)

	void DownloadAndSave_8Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	cdatabuf_8chan

	void SaveMat_1chan(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	mat[i] in DownloadAndSave_8Channel(..)
# 8 channels, 1 map


	void DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, bool exception_tiff=false );
	cdatabuf, hdatabuf
# 1 channel, many maps
 *
 */
