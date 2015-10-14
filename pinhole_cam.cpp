/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-14
*/

#include <iostream>
#include "./pinhole_cam.hpp"

using namespace vr;

PinholeCamera::PinholeCamera(const json & options): Map(options) {
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

std::vector<PointAndFlag> PinholeCamera::
    lonlat_to_xy_batch(const std::vector<std::pair<double, double>> & points) {

    std::vector<std::tuple<double, double, bool>> ret;
    ret.resize(points.size());

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    std::vector<int> index;

    for(int i = 0 ; i < points.size() ; i += 1) {
        auto & point = points[i];
        double lon = -point.first;
        double lat = -point.second;
        if(lon < 0) {
            ret[i] = std::make_tuple(0, 0, false);
            continue;
        }
        double x = cos(lon) * cos(lat);
        double z = sin(lon) * cos(lat);
        double y = sin(lat);
        objectPoints.push_back(cv::Point3f(x, y, z));
        index.push_back(i);
    }

    cv::projectPoints(objectPoints, 
                      cv::Mat::zeros(1, 3, CV_64F), cv::Mat::zeros(1, 3, CV_64F),
                      camera_matrix, dist_coeffs, imagePoints);

    for(int i = 0 ; i < index.size() ; i += 1) {
        cv::Point2f image_p = imagePoints[i];
        ret[index[i]] = std::make_tuple(image_p.x / this->width, 
                                        image_p.y / this->height, true);
    }

    return ret;
}
