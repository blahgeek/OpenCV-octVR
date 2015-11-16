/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-16
*/

#ifndef VR_LIBMAP_IMPL_H
#define VR_LIBMAP_IMPL_H value

#include "./camera.hpp"
#include "./libmap.hpp"
#include <opencv2/stitching/detail/exposure_compensate.hpp>

#include <opencv2/core/cuda.hpp>

namespace vr {

using cv::cuda::GpuMat;

class MultiMapperImpl: public MultiMapper {
private:
    cv::Size out_size;
    std::vector<cv::Size> in_sizes;

    std::vector<GpuMat> map1s; // CV_32FC1
    std::vector<GpuMat> map2s;
    std::vector<GpuMat> masks; 

    std::vector<cv::Point2d> output_map_points;

private:
    std::vector<double> working_scales;
    std::vector<GpuMat> scaled_masks;

private:
    cv::Ptr<cv::detail::ExposureCompensator> compensator;
    std::vector<GpuMat> seam_masks;

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

    void prepare() override;

    void get_output(const std::vector<cv::Mat> & inputs, cv::Mat & output) override;

    void get_single_output(const cv::Mat & input, cv::Mat & output) override;

    void reset_compensator() override {
        compensator.release();
    }

    void dump(std::ofstream & f) override;
};

}

#endif
