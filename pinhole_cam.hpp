/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-14
*/

#ifndef VR_LIBMAP_PINHOLE_CAM_H
#define VR_LIBMAP_PINHOLE_CAM_H value

#include "./libmap.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <string>
#include <vector>

namespace vr {

class PinholeCamera: public Map {
private:
    cv::Mat camera_matrix;
    std::vector<double> dist_coeffs;
    int width, height;
public:
    PinholeCamera(const json & options);
    double get_aspect_ratio() override {
        return double(width) / double(height);
    }
    std::pair<double, double> lonlat_to_xy(double lon, double lat) override;
};

}

#endif
