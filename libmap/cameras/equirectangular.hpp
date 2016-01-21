/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-08
*/

#ifndef VR_LIBMAP_EQUIRECTANGULAR_H
#define VR_LIBMAP_EQUIRECTANGULAR_H value

#include "../camera.hpp"
#include "../libmap_impl.hpp"

namespace vr {

/**
 * 2:1 equirectangular map
 * left-top corner: longitude -PI, latitude PI/2
 * center: longitude 0, latitude 0
 *
 * Options: 
 *     min_lat/max_lat: default to -PI/2, PI/2
 *     scale_lon: default to 1.0
 */
class Equirectangular: public Camera {
private:
    double min_lat = -M_PI/2, max_lat = M_PI/2;
    double scale_lon = 1.0;
public:
    // using Camera::Camera;
    Equirectangular(const json & options);
    
    virtual double get_aspect_ratio() override {
        return (2.0f * scale_lon) / ((max_lat - min_lat) / M_PI);
    }

    cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override;
    cv::Point2d image_to_obj_single(const cv::Point2d & xy) override;
};

}

#endif
