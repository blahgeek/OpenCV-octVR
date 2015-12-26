/* 
* @Author: BlahGeek
* @Date:   2015-11-12
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-12
*/

#ifndef VR_LIBMAP_STUPIDOVAL_H_
#define VR_LIBMAP_STUPIDOVAL_H_ value

#include "../camera.hpp"
#include <math.h>

namespace vr {

class StupidOval: public Camera {
public:
    using Camera::Camera;

    virtual double get_aspect_ratio() override {
        return 2.0;
    }
    cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override {
        double x = cos(lonlat.y) * lonlat.x / (M_PI * 2.0) + 0.5;
        double y = - lonlat.y / M_PI + 0.5;

        return cv::Point2d(x, y);
    }
    cv::Point2d image_to_obj_single(const cv::Point2d & xy) override {
        double lat = (0.5 - xy.y) * M_PI;
        double lon = (xy.x - 0.5) * M_PI * 2.0 / cos(lat);
        if(lon < -M_PI || lon > M_PI)
            return cv::Point2d(NAN, NAN);
        return cv::Point2d(lon, lat);
    }
};

}

#endif
