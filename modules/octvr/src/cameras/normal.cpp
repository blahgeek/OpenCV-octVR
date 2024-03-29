/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-14
*/

#include <iostream>
#include "./normal.hpp"

using namespace vr;

Normal::Normal(const rapidjson::Value & options): Camera(options) {
    this->aspect_ratio = options["aspect_ratio"].GetDouble();

    cam_x = options["cam_opt"].GetDouble();
    cam_z = sqrt((1.0 - cam_x * cam_x) / 
                 (1.0 + 1.0 / this->aspect_ratio / this->aspect_ratio));
    cam_y = cam_z / this->aspect_ratio;

    std::cerr << "Normal camera: " << cam_x << "/" << aspect_ratio << std::endl;
}

cv::Point2d Normal::image_to_obj_single(const cv::Point2d & xy) {
    double xx = cam_x;
    double yy = cam_y - xy.y * 2.0 * cam_y;
    double zz = cam_z - xy.x * 2.0 * cam_z;

    return sphere_xyz_to_lonlat(cv::Point3d(xx, yy, zz));
}

cv::Point2d Normal::obj_to_image_single(const cv::Point2d & lonlat) {
    cv::Point3d xxyyzz = sphere_lonlat_to_xyz(lonlat);
    if(xxyyzz.x < 0)
        return cv::Point2d(NAN, NAN);
    xxyyzz /= (xxyyzz.x / cam_x);
    return cv::Point2d((cam_z - xxyyzz.z) / 2.0 / cam_z,
                       (cam_y - xxyyzz.y) / 2.0 / cam_y);
}

