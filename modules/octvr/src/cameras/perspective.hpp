/* 
* @Author: BlahGeek
* @Date:   2016-03-15
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-15
*/

#ifndef VR_LIBMAP_PERSPECTIVE_H
#define VR_LIBMAP_PERSPECTIVE_H value

#include "../camera.hpp"

namespace vr {

// 参数同ocam代码中的create_perspective_undistortion

    class PerspectiveCamera: public Camera {
    private:
        double sf;
        double aspect_ratio;

    public:
        PerspectiveCamera(const rapidjson::Value & options);
        double get_aspect_ratio() override {
            return this->aspect_ratio;
        }

        cv::Point2d image_to_obj_single(const cv::Point2d & xy) override;
        cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override;
    };

}

#endif
