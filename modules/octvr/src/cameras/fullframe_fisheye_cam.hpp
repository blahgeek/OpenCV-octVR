/* 
* @Author: BlahGeek
* @Date:   2015-11-03
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-21
*/

#ifndef VR_LIBMAP_FULLFRAME_FISHEYE_H_
#define VR_LIBMAP_FULLFRAME_FISHEYE_H_ value

#include "../camera.hpp"

namespace vr {

class FullFrameFisheyeCamera: public Camera {
protected:
    cv::Rect circular_crop;
    double radial_distortion[6];
    double hfov; // horizon fov
    cv::Point2d center_shift; // 0-1
    cv::Size size;

protected:
    cv::Point2d do_radial_distort(cv::Point2d src);

public:
    FullFrameFisheyeCamera(const rapidjson::Value & options);

    double get_aspect_ratio() override;
    cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override;
};

}

#endif
