/* 
* @Author: BlahGeek
* @Date:   2015-10-14
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-22
*/

#include "./fisheye_cam.hpp"

using namespace vr;

// TODO: OpenCV的鱼眼模型无法处理覆盖大于二分之一个球面的情况
void FisheyeCamera::_project(std::vector<cv::Point3d> & objectPoints,
                             std::vector<cv::Point2d> & imagePoints) {
    cv::fisheye::projectPoints(objectPoints, imagePoints,
                               cv::Mat::zeros(1, 3, CV_64F), cv::Mat::zeros(1, 3, CV_64F),
                               camera_matrix, dist_coeffs);
}
