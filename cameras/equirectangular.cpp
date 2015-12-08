/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-08
*/

#include "./equirectangular.hpp"
#include <stdio.h>

using namespace vr;

Equirectangular::Equirectangular(const json & options): Camera(options) {
    #define ASSIGN(KEY) \
        if(options.find( #KEY ) != options.end()) \
            this->KEY = options[#KEY].get<double>();
    ASSIGN(min_lat)
    ASSIGN(max_lat)
    ASSIGN(scale_lon)

    fprintf(stderr, "Equirectangular, lat = [%f, %f], lon scale = %f\n",
            this->min_lat, this->max_lat, this->scale_lon);
}

cv::Point2d Equirectangular::obj_to_image_single(const cv::Point2d & lonlat) {
    double x = lonlat.x / (M_PI * 2.0) + 0.5;
    // double y = - lonlat.y / M_PI + 0.5;
    double y = (-lonlat.y - this->min_lat) / (this->max_lat - this->min_lat);
    return cv::Point2d(x, y);
}

cv::Point2d Equirectangular::image_to_obj_single(const cv::Point2d & xy) {
    double lon = (xy.x - 0.5) * M_PI * 2.0;
    // double lat = (0.5 - xy.y) * M_PI;
    double lat = - xy.y * (this->max_lat - this->min_lat) - this->min_lat;
    return cv::Point2d(lon, lat);
}
