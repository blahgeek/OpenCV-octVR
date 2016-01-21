/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-22
*/

#include <iostream>
#include "./pinhole_cam.hpp"

using namespace vr;

PinholeCamera::PinholeCamera(const json & options): Camera(options) {
    this->camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    this->camera_matrix.at<double>(0, 0) = options["fx"];
    this->camera_matrix.at<double>(1, 1) = options["fy"];
    this->camera_matrix.at<double>(0, 2) = options["cx"];
    this->camera_matrix.at<double>(1, 2) = options["cy"];

    this->dist_coeffs = options["dist_coeffs"].get<std::vector<double>>();
    this->width = options["width"].get<int>();
    this->height = options["height"].get<int>();

    std::cerr << "Camera matrix: " << this->camera_matrix << std::endl;
    std::cerr << "Distort coeffs: ";
    for(auto x: dist_coeffs) std::cerr << x << ", ";
    std::cerr << std::endl;
    std::cerr << "Camera size: " << width << " x " << height << std::endl;
}

std::vector<cv::Point2d> PinholeCamera::obj_to_image(const std::vector<cv::Point2d> & lonlats) {
    std::vector<cv::Point3d> objectPoints;
    for(auto & lonlat: lonlats)
        objectPoints.push_back(sphere_lonlat_to_xyz(lonlat));
    this->sphere_rotate(objectPoints, false);

    for(auto & p: objectPoints)
        if(p.z <= 0)
            p.x = p.y = p.z = NAN;

    std::vector<cv::Point2d> imagePoints;
    this->_project(objectPoints, imagePoints);

    std::vector<cv::Point2d> ret;
    ret.reserve(imagePoints.size());
    for(auto & p: imagePoints)
        ret.push_back(cv::Point2d(p.x / this->width, 1.0 - p.y / this->height));
    return ret;
}

void PinholeCamera::_project(std::vector<cv::Point3d> & objectPoints,
                             std::vector<cv::Point2d> & imagePoints) {
    cv::projectPoints(objectPoints, 
                      cv::Mat::zeros(1, 3, CV_64F), cv::Mat::zeros(1, 3, CV_64F),
                      camera_matrix, dist_coeffs, imagePoints);
}
