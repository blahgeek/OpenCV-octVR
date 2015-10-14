/* 
* @Author: BlahGeek
* @Date:   2015-10-14
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-14
*/

#ifndef VR_LIBMAP_FISHEYE_H
#define VR_LIBMAP_FISHEYE_H value

#include "./pinhole_cam.hpp"

namespace vr {

class FisheyeCamera: public PinholeCamera {
public:
    using PinholeCamera::PinholeCamera;
protected:
    void _project(std::vector<cv::Point3f> & objectPoints,
                  std::vector<cv::Point2f> & imagePoints) override;
};

}

#endif
