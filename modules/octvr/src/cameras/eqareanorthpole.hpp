
#ifndef VR_LIBMAP_EQAREANORTHPOLE_H_
#define VR_LIBMAP_EQAREANORTHPOLE_H_ value
#include "../camera.hpp"
#include <iostream>

namespace vr {

class Eqareanorthpole: public Camera {
private:
    double arctic_circle = M_PI / 3;
public:
    //using Camera::Camera;
    Eqareanorthpole(const rapidjson::Value & options) :Camera(options){
        if(options.HasMember("arctic_circle"))
            this->arctic_circle = options["arctic_circle"].GetDouble();
        std::cerr << "Eqareanorthpole with arctic_circle = " << this->arctic_circle << std::endl;
    }

    double get_aspect_ratio() override {
        return 1.0;
    }

    cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override {
        if (lonlat.y < arctic_circle)
            return cv::Point2d(NAN, NAN);
        else {
            double rho = (M_PI / 2 - lonlat.y) / (M_PI / 2 - arctic_circle);
            double x = - rho * sin(lonlat.x) / 2 + 0.5;
            double y = - rho * cos(lonlat.x) / 2 + 0.5;
            return cv::Point2d(x,y);
        }
    }

    cv::Point2d image_to_obj_single(const cv::Point2d & xy) override {
        cv::Point2d diff = xy - cv::Point2d(0.5, 0.5);
        double rho = cv::sqrt(diff.x * diff.x + diff.y * diff.y) * 2;
        double lat = M_PI / 2 - (M_PI / 2 - arctic_circle) * rho;
        double lon = atan2(-diff.x, -diff.y);
        return cv::Point2d(lon, lat);
    }

};

}

#endif
