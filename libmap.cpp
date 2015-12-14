/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-07
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

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

// TODO Set by options
#define WORKING_MEGAPIX 0.1

using namespace vr;

Mapper::Mapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes, int blend) {
    Timer timer("Mapper constructor");

    std::vector<GpuMat> scaled_masks;

    this->out_size = mt.out_size;
    this->map1s.resize(mt.map1s.size());
    this->map2s.resize(mt.map1s.size());
    this->masks.resize(mt.map1s.size());
    this->seam_masks.resize(mt.map1s.size());
    scaled_masks.resize(mt.map1s.size());

    this->working_scale = std::min(1.0, sqrt(WORKING_MEGAPIX * 1e6 / out_size.area()));

    for(int i = 0 ; i < mt.map1s.size() ; i += 1) {
        map1s[i].upload(mt.map1s[i]);
        map2s[i].upload(mt.map2s[i]);
        masks[i].upload(mt.masks[i]);
        seam_masks[i].upload(mt.seam_masks[i]);
        cv::cuda::resize(masks[i], scaled_masks[i],
                         cv::Size(), working_scale, working_scale);
    }
    timer.tick("Uploading mats");

    this->streams.resize(masks.size());

    this->gpu_inputs.resize(masks.size());
    this->warped_imgs.resize(masks.size());
    this->warped_imgs_scale.resize(masks.size());
    for(int i = 0 ; i < masks.size() ; i += 1) {
        gpu_inputs[i].create(in_sizes[i], CV_8UC4);
        warped_imgs[i].create(out_size, CV_8UC4);
        cv::cuda::resize(warped_imgs[i], warped_imgs_scale[i],
                         cv::Size(), working_scale, working_scale,
                         cv::INTER_NEAREST);
    }
    this->result.create(out_size, CV_8UC3);

    if(blend == 0)
        return;

    timer.tick("Allocating internal mats");

    if(blend > 0) {
        int blend_bands = int(ceil(log(blend)/log(2.)) - 1.);
        std::cerr << "Using MultiBandBlender with band number = " << blend_bands << std::endl;
        this->blender = cv::makePtr<cv::detail::MultiBandGPUBlender>(seam_masks, blend_bands);
    } else {
        //float sharpness = 1.0 / float(-blend);
        std::cerr << "Using FeatherBlender with border = " << -blend << std::endl;
        this->blender = cv::makePtr<cv::detail::FeatherGPUBlender>(masks, -blend);
    }

    timer.tick("Blender initialize");

    this->compensator = cv::makePtr<cv::detail::GainCompensatorGPU>(scaled_masks);
    timer.tick("Gain Compensator initialize");

}

void Mapper::stitch(const std::vector<GpuMat> & inputs,
                        GpuMat & output) {
    Timer timer("Stitch");

    assert(inputs.size() == masks.size());
    for(int i = 0 ; i < inputs.size() ; i += 1)
        assert(inputs[i].type() == CV_8UC3); // RGB
    assert(output.type() == CV_8UC3 && output.size() == this->out_size);
    
    std::vector<cv::Point2i> corners;
    
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        cv::cuda::cvtColor(inputs[i], gpu_inputs[i], cv::COLOR_RGB2RGBA, 4, streams[i]);
        cv::cuda::fastRemap(gpu_inputs[i], warped_imgs[i], map1s[i], map2s[i], true, streams[i]);
        cv::cuda::resize(warped_imgs[i], warped_imgs_scale[i],
                         cv::Size(), working_scale, working_scale,
                         cv::INTER_NEAREST, streams[i]);
        corners.emplace_back(0, 0);
    }

    for(auto & s: streams)
        s.waitForCompletion();
    timer.tick("Uploading and remapping and resizing images");

    compensator->feed(warped_imgs_scale);
    timer.tick("Compensator");

    compensator->apply(warped_imgs, masks);
    timer.tick("Compensator apply");

    blender->blend(warped_imgs, output);
    timer.tick("Blender blend");

    assert(output.type() == CV_8UC3);
}

void Mapper::remap(const std::vector<GpuMat> & inputs,
                       GpuMat & output) {
    Timer timer("Remap");

    assert(inputs.size() == masks.size());
    for(int i = 0 ; i < inputs.size() ; i += 1)
        assert(inputs[i].type() == CV_8UC3); // RGB

    assert(output.type() == CV_8UC3);
    assert(output.size() == this->out_size);
    
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        cv::cuda::cvtColor(inputs[i], gpu_inputs[i], cv::COLOR_RGB2RGBA, 4, streams[i]);
        cv::cuda::fastRemap(gpu_inputs[i], warped_imgs[0], map1s[i], map2s[i], false, streams[i]);
    }

    for(auto & s: streams)
        s.waitForCompletion();
    timer.tick("Remapping images");

    cv::cuda::cvtColor(warped_imgs[0], output, cv::COLOR_RGBA2RGB, 3, streams[0]);
    streams[0].waitForCompletion();

    timer.tick("Convert");

}
