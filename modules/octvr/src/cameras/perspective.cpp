/* 
* @Author: BlahGeek
* @Date:   2016-03-15
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-15
*/

#include <iostream>

#include "./perspective.hpp"

using namespace vr;

PerspectiveCamera::PerspectiveCamera(const rapidjson::Value & options): Camera(options) {
    this->aspect_ratio = options["aspect_ratio"].GetDouble();
    this->sf = options["sf"].GetDouble();

    std::cerr << "Perspective camera with sf=" << this->sf << std::endl;
}

cv::Point2d PerspectiveCamera::image_to_obj_single(const cv::Point2d & xy) {
    double z = (0.5 - xy.x) * this->aspect_ratio;
    double y = 0.5 - xy.y;
    double x = 1.0 / this->sf;
    return sphere_xyz_to_lonlat(cv::Point3d(x, y, z));
}

cv::Point2d PerspectiveCamera::obj_to_image_single(const cv::Point2d & lonlat) {
    cv::Point3d xyz = sphere_lonlat_to_xyz(lonlat);
    double y_ = xyz.y * (1.0 / this->sf / xyz.x);
    double z_ = xyz.z * (1.0 / this->sf / xyz.x);
    return cv::Point2d(0.5 - z_ / this->aspect_ratio, 0.5 - y_);
}
