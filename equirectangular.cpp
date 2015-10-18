/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-18
*/

#include "./equirectangular.hpp"

using namespace vr;

cv::Point2d Equirectangular::obj_to_image_single(const cv::Point2d & lonlat) {
    double x = lonlat.x / (M_PI * 2.0) + 0.5;
    double y = - lonlat.y / M_PI + 0.5;
    return cv::Point2d(x, y);
}

cv::Point2d Equirectangular::image_to_obj_single(const cv::Point2d & xy) {
    double lon = (xy.x - 0.5) * M_PI * 2.0;
    double lat = (0.5 - xy.y) * M_PI;
    return cv::Point2d(lon, lat);
}
