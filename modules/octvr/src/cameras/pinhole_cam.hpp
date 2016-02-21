/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-21
*/

#ifndef VR_LIBMAP_PINHOLE_CAM_H
#define VR_LIBMAP_PINHOLE_CAM_H value

#include "../camera.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <string>
#include <vector>

namespace vr {

class PinholeCamera: public Camera {
protected:
    cv::Mat camera_matrix;
    std::vector<double> dist_coeffs;
    int width, height;

    virtual void _project(std::vector<cv::Point3d> & objectPoints,
                          std::vector<cv::Point2d> & imagePoints);
public:
    PinholeCamera(const rapidjson::Value & options);
    double get_aspect_ratio() override {
        return double(width) / double(height);
    }
    std::vector<cv::Point2d> obj_to_image(const std::vector<cv::Point2d> & lonlats) override;
};

}

#endif
