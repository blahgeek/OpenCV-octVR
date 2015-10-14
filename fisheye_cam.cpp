/* 
* @Author: BlahGeek
* @Date:   2015-10-14
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-14
*/

#include "./fisheye_cam.hpp"

using namespace vr;

void FisheyeCamera::_project(std::vector<cv::Point3f> & objectPoints,
                             std::vector<cv::Point2f> & imagePoints) {
    cv::fisheye::projectPoints(objectPoints, imagePoints,
                               cv::Mat::zeros(1, 3, CV_64F), cv::Mat::zeros(1, 3, CV_64F),
                               camera_matrix, dist_coeffs);
}
