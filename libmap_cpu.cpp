/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-16
*/

#include <iostream>
#include "./libmap_impl.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <assert.h>

#include <opencv2/core/cuda.hpp>

#ifdef HAVE_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

using namespace vr;

CPUMapper::CPUMapper(const MapperTemplate & mt, 
                     std::vector<cv::Size> in_sizes) {
    Timer timer("Mapper constructor");

    assert(mt.overlay_inputs.size() == 0);

    this->out_size = mt.out_size;

    for(int i = 0 ; i < mt.inputs.size() ; i += 1) {
        auto & in = mt.inputs[i];

        cv::UMat map1, map2;
        cv::convertMaps(in.map1 * in_sizes[i].width, 
                        in.map2 * in_sizes[i].height, 
                        map1, map2, CV_16SC2);
        this->map1s.push_back(map1);
        this->map2s.push_back(map2);
        cv::UMat mask;
        in.mask.copyTo(mask);
        this->masks.push_back(mask);
    }

    timer.tick("Copying maps and masks");

    std::vector<cv::Point> corners(mt.inputs.size(), cv::Point(0, 0));
    std::vector<cv::UMat> weight_maps;
    cv::detail::FeatherBlender blender;
    blender.createWeightMaps(this->masks,
                             corners,
                             weight_maps);
    this->feather_masks.resize(weight_maps.size());
    for(int i = 0 ; i < weight_maps.size() ; i += 1)
        weight_maps[i].convertTo(this->feather_masks[i], CV_8U, 255.0);

    timer.tick("Prepare feather masks");
}

void CPUMapper::stitch(const std::vector<UMat> & inputs, UMat & output) {
    Timer timer("stitch");

    output.create(this->out_size, CV_8UC3);
    output.setTo(0);

    std::vector<UMat> output_channels(3);
    for(int k = 0 ; k < 3 ; k += 1)
        output_channels[k].create(this->out_size, CV_32F);

    for(int i = 0 ; i < this->masks.size() ; i += 1) {
        assert(inputs[i].type() == CV_8UC3);

        std::vector<UMat> input_channels(3);
        std::vector<UMat> remapped_channels(3);

        cv::split(inputs[i], input_channels);
        timer.tick("Split input");

        for(int k = 0 ; k < 3 ; k += 1)
            cv::remap(input_channels[k], 
                      remapped_channels[k],
                      this->map1s[i], this->map2s[i],
                      cv::INTER_LINEAR);
        timer.tick("remap");

        for(int k = 0 ; k < 3 ; k += 1)
            cv::accumulateProduct(remapped_channels[k],
                                  this->feather_masks[i],
                                  output_channels[k]); // TODO: mask
        timer.tick("accumulate product");
    }

    UMat output_f;
    cv::merge(output_channels, output_f);
    output_f.convertTo(output, CV_8UC3, 1.0/255.0);

}
