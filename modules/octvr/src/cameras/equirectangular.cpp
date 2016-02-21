/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-21
*/

#include "./equirectangular.hpp"
#include <stdio.h>

using namespace vr;

Equirectangular::Equirectangular(const rapidjson::Value & options): Camera(options) {
    #define ASSIGN(KEY) \
        if(options.HasMember( #KEY )) \
            this->KEY = options[#KEY].GetDouble();
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
