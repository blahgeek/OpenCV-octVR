/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-23
*/

#ifndef VR_LIBMAP_IMPL_H
#define VR_LIBMAP_IMPL_H value

#include "./camera.hpp"
#include "./libmap.hpp"

namespace vr {

class MultiMapperImpl: public MultiMapper {
private:
    std::unique_ptr<Camera> out_camera;
    std::vector<std::unique_ptr<Camera>> in_cameras;

    cv::Size out_size;
    std::vector<cv::Size> in_sizes;

    std::vector<cv::Mat> map1s;
    std::vector<cv::Mat> map2s; // see opencv: convertMaps()
    std::vector<cv::Mat> masks; 

    std::vector<cv::Point2d> output_map_points;

public:
    MultiMapperImpl(const std::string & to, const json & to_opts, 
                    int out_width, int out_height);
    void add_input(const std::string & from, const json & from_opts,
                   int in_width, int in_height);

    cv::Size get_output_size() {
        return this->out_size;
    }

    void get_output(const std::vector<cv::Mat> & inputs, cv::Mat & output);
};

}

#endif
