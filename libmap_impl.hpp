/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-12
*/

#ifndef VR_LIBMAP_IMPL_H
#define VR_LIBMAP_IMPL_H value

#include "./camera.hpp"
#include "./libmap.hpp"

namespace vr {

class MultiMapperImpl: public MultiMapper {
private:
    cv::Size out_size;
    std::vector<cv::Size> in_sizes;

    std::vector<cv::UMat> map1s;
    std::vector<cv::UMat> map2s; // see opencv: convertMaps()
    std::vector<cv::UMat> masks; 

    std::vector<cv::Point2d> output_map_points;

public:
    MultiMapperImpl(const std::string & to, const json & to_opts, 
                    int out_width, int out_height);
    explicit MultiMapperImpl(std::ifstream & f);

    void add_input(const std::string & from, const json & from_opts,
                   int in_width, int in_height) override;

    cv::Size get_output_size() override {
        return this->out_size;
    }
    cv::Size get_input_size(int index) override {
        if(index >= in_sizes.size())
            return cv::Size(0, 0);
        return in_sizes[index];
    }

    void get_output(const std::vector<cv::UMat> & inputs, cv::UMat & output) override;

    void get_single_output(const cv::UMat & input, cv::UMat & output) override;

    void dump(std::ofstream & f) override;
};

}

#endif
