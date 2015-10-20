/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-20
*/

#ifndef VR_LIBMAP_EQUIRECTANGULAR_H
#define VR_LIBMAP_EQUIRECTANGULAR_H value

#include "./camera.hpp"
#include "./libmap_impl.hpp"

namespace vr {

/**
 * 2:1 equirectangular map
 * left-top corner: longitude -PI, latitude PI/2
 * center: longitude 0, latitude 0
 */
class Equirectangular: public Camera {
public:
    using Camera::Camera;
    
    virtual double get_aspect_ratio() override {
        return 2.0f;
    }

    cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override;
    cv::Point2d image_to_obj_single(const cv::Point2d & xy) override;
};

}

#endif
