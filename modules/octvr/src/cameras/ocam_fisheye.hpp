/* 
* @Author: BlahGeek
* @Date:   2016-03-10
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-10
*/

#ifndef OCTVR_CAMERAS_OCAM_FISHEYE_H__
#define OCTVR_CAMERAS_OCAM_FISHEYE_H__ value

#include "../camera.hpp"

namespace vr {

class OCamFisheyeCamera: public Camera {


public:
// from ocam_functions.h

#define CMV_MAX_BUF 1024
#define MAX_POL_LENGTH 64

    struct ocam_model {
        double pol[MAX_POL_LENGTH];    // the polynomial coefficients: pol[0] + x"pol[1] + x^2*pol[2] + ... + x^(N-1)*pol[N-1]
        int length_pol;                // length of polynomial
        double invpol[MAX_POL_LENGTH]; // the coefficients of the inverse polynomial
        int length_invpol;             // length of inverse polynomial
        double xc;         // row coordinate of the center
        double yc;         // column coordinate of the center
        double c;          // affine parameter
        double d;          // affine parameter
        double e;          // affine parameter
        int width;         // image width
        int height;        // image height
    };
    
private:
    struct ocam_model model;

public:
    OCamFisheyeCamera(const rapidjson::Value & options);

    double get_aspect_ratio() override;
    cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override;
    cv::Point2d image_to_obj_single(const cv::Point2d & xy) override;
};

}

#endif
