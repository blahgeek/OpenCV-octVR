/* 
* @Author: BlahGeek
* @Date:   2015-10-14
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-18
*/

#include "./fisheye_cam.hpp"

using namespace vr;

void FisheyeCamera::_project(std::vector<cv::Point3d> & objectPoints,
                             std::vector<cv::Point2d> & imagePoints) {
    cv::fisheye::projectPoints(objectPoints, imagePoints,
                               this->rotate_vector, cv::Mat::zeros(1, 3, CV_64F),
                               camera_matrix, dist_coeffs);
}
