
#ifndef VR_LIBMAP_EQAREASOUTHPOLE_H_
#define VR_LIBMAP_EQAREASOUTHPOLE_H_ value
#include "../camera.hpp"

namespace vr {

class Eqareasouthpole: public Camera {
private:
    double antarctic_circle = - M_PI / 3;
public:
    //using Camera::Camera;
    Eqareasouthpole(const rapidjson::Value & options) :Camera(options){
        if(options.HasMember("antarctic_circle"))
            this->antarctic_circle = options["antarctic_circle"].GetDouble();
        std::cerr << "Eqareasouthpole with antarctic_circle = " << this->antarctic_circle << std::endl;
    }

    double get_aspect_ratio() override {
        return 1.0;
    }

    cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override {
        if (lonlat.y > antarctic_circle)
            return cv::Point2d(NAN, NAN);
        else {
            double rho = (lonlat.y + M_PI / 2) / (antarctic_circle + M_PI / 2);
            double x = rho * sin(lonlat.x) / 2 + 0.5;
            double y = - rho * cos(lonlat.x) / 2 + 0.5;
            return cv::Point2d(x,y);
        }
    }

    cv::Point2d image_to_obj_single(const cv::Point2d & xy) override {
        cv::Point2d diff = xy - cv::Point2d(0.5, 0.5);
        double rho = cv::sqrt(diff.x * diff.x + diff.y * diff.y) * 2;
        double lat = - M_PI / 2 + (antarctic_circle + M_PI / 2) * rho;
        double lon = atan2(diff.x, -diff.y);
        return cv::Point2d(lon, lat);
    }
};

}

#endif
