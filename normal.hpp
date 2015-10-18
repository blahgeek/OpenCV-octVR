/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-18
*/

#ifndef VR_LIBMAP_NORMAL_H
#define VR_LIBMAP_NORMAL_H value

#include "./libmap.hpp"

namespace vr {

/**
 * Simplified pinhole camera model
 * left-top corner: lon -, lat +
 * center: lon 0, lat 0
 */
class Normal: public Map {
private:
    double cam_x, cam_y, cam_z;
    double aspect_ratio;
public:
    Normal(const json & options);
    double get_aspect_ratio() override {
        return this->aspect_ratio;
    }

    cv::Point2d image_to_obj_single(const cv::Point2d & xy) override;

};

}

#endif
