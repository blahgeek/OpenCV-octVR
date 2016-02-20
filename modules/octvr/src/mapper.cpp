/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-20
*/

#include <iostream>
#include "./mapper.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <assert.h>

#include "opencv2/core/cuda.hpp"

#include "cvconfig.h"

#ifdef HAVE_CUDA
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#endif

// TODO Set by options
#define WORKING_MEGAPIX 0.1

using namespace vr;

Mapper::Mapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes, 
               int blend, bool enable_gain_compensator,
               cv::Size scale_output) {
#ifdef WITH_DONGLE_LICENSE
    lic_runtime_init(&(this->lic_t), 601);
    this->lic_cnt = 0;
#endif

#ifdef HAVE_CUDA
    Timer timer("Mapper constructor");

    std::vector<GpuMat> scaled_masks;

    this->nonoverlay_num = mt.inputs.size();

    this->stitch_size = mt.out_size;
    this->scaled_output_size = scale_output.area() == 0 ? mt.out_size : scale_output;

    this->map1s.resize(mt.inputs.size() + mt.overlay_inputs.size());
    this->map2s.resize(mt.inputs.size() + mt.overlay_inputs.size());
    this->masks.resize(mt.inputs.size() + mt.overlay_inputs.size());

    this->seam_masks.resize(mt.inputs.size());
    scaled_masks.resize(mt.inputs.size());

    this->working_scale = std::min(1.0, sqrt(WORKING_MEGAPIX * 1e6 / stitch_size.area()));

    for(int i = 0 ; i < nonoverlay_num ; i += 1) {
        map1s[i].upload(mt.inputs[i].map1);
        map2s[i].upload(mt.inputs[i].map2);
        masks[i].upload(mt.inputs[i].mask);
        seam_masks[i].upload(mt.seam_masks[i]);
        if(enable_gain_compensator)
            cv::cuda::resize(masks[i], scaled_masks[i],
                             cv::Size(), working_scale, working_scale);
    }
    for(int i = 0 ; i < mt.overlay_inputs.size() ; i += 1) {
        map1s[i + nonoverlay_num].upload(mt.overlay_inputs[i].map1);
        map2s[i + nonoverlay_num].upload(mt.overlay_inputs[i].map2);
        masks[i + nonoverlay_num].upload(mt.overlay_inputs[i].mask);
    }

    timer.tick("Uploading mats");

    this->streams.resize(mt.inputs.size() + mt.overlay_inputs.size() + 1);
    this->rgb_inputs.resize(mt.inputs.size() + mt.overlay_inputs.size());
    this->rgba_inputs.resize(mt.inputs.size() + mt.overlay_inputs.size());
    this->warped_imgs.resize(mt.inputs.size() + mt.overlay_inputs.size());

    this->warped_imgs_scale.resize(mt.inputs.size() + mt.overlay_inputs.size());
    for(int i = 0 ; i < mt.inputs.size() ; i += 1) {
        rgb_inputs[i].create(in_sizes[i], CV_8UC3);
        rgba_inputs[i].create(in_sizes[i], CV_8UC4);
        warped_imgs[i].create(stitch_size, CV_8UC4);
        if(enable_gain_compensator)
            cv::cuda::resize(warped_imgs[i], warped_imgs_scale[i],
                             cv::Size(), working_scale, working_scale,
                             cv::INTER_NEAREST);
    }

    for(int i = mt.inputs.size() ; i < mt.inputs.size() + mt.overlay_inputs.size() ; i += 1) {
        rgb_inputs[i].create(in_sizes[i], CV_8UC3);
        rgba_inputs[i].create(in_sizes[i], CV_8UC4);
        warped_imgs[i].create(stitch_size, CV_8UC4);
        warped_imgs_scale[i].create(stitch_size, CV_8UC3);  // NOTICE
    }

    this->result.create(stitch_size, CV_8UC3);
    if(this->stitch_size != this->scaled_output_size)
        this->result_scaled.create(this->scaled_output_size, CV_8UC3);

    timer.tick("Allocating internal mats");

    if(blend > 0) {
        int blend_bands = int(ceil(log(blend)/log(2.)) - 1.);
        std::cerr << "Using MultiBandBlender with band number = " << blend_bands << std::endl;
        this->blender.reset(new cv::detail::MultiBandGPUBlender(seam_masks, blend_bands));
    } else if(blend < 0) {
        //float sharpness = 1.0 / float(-blend);
        std::cerr << "Using FeatherBlender with border = " << -blend << std::endl;
        this->blender.reset(new cv::detail::FeatherGPUBlender(std::vector<GpuMat>(masks.begin(), masks.begin() + nonoverlay_num), 
                                                              -blend));
    } else {
        std::cerr << "Do not use blender" << std::endl;
        this->blender = nullptr;
    }

    timer.tick("Blender initialize");

    if(enable_gain_compensator)
        this->compensator.reset(new cv::detail::GainCompensatorGPU(scaled_masks));
    else {
        std::cerr << "Do not enable gain compensator" << std::endl;
        this->compensator = nullptr;
    }
    timer.tick("Gain Compensator initialize");
#else
    assert(false);
#endif

}

void Mapper::stitch(std::vector<GpuMat> & inputs,
                    GpuMat & output) {
#ifdef WITH_DONGLE_LICENSE
    this->lic_cnt += 1;
    if (this->lic_cnt % 3000 == 0) {
        lic_runtime_check(&(this->lic_t));
    }
#endif

#ifdef HAVE_CUDA
    Timer timer("Stitch");

    assert(inputs.size() == masks.size());
    for(int i = 0 ; i < inputs.size() ; i += 1)
        assert(inputs[i].type() == CV_8UC2); // UYVY422
    assert(output.type() == CV_8UC2 && output.size() == this->scaled_output_size);

    int swap_orders[] = {3, 2, 1, 0};
    
    for(int i = 0 ; i < nonoverlay_num ; i += 1) {
        cv::cuda::GpuMat input_c4 = inputs[i].reshape(4);
        cv::cuda::swapChannels(input_c4, swap_orders, streams[i]);
        cv::cuda::cvtUYVY422toRGB24(inputs[i], rgb_inputs[i], streams[i]);
        cv::cuda::cvtColor(rgb_inputs[i], rgba_inputs[i], cv::COLOR_RGB2RGBA, 4, streams[i]);
        cv::cuda::fastRemap(rgba_inputs[i], warped_imgs[i], map1s[i], map2s[i], true, streams[i]);
        if(this->compensator)
            cv::cuda::resize(warped_imgs[i], warped_imgs_scale[i],
                             cv::Size(), working_scale, working_scale,
                             cv::INTER_NEAREST, streams[i]);
    }

    for(int i = nonoverlay_num ; i < inputs.size() ; i += 1) {
        cv::cuda::cvtUYVY422toRGB24(inputs[i], rgb_inputs[i], streams[i]);
        cv::cuda::cvtColor(rgb_inputs[i], rgba_inputs[i], cv::COLOR_RGB2RGBA, 4, streams[i]);
        cv::cuda::fastRemap(rgba_inputs[i], warped_imgs[i], map1s[i], map2s[i], false, streams[i]);
        cv::cuda::cvtColor(warped_imgs[i], warped_imgs_scale[i], cv::COLOR_RGBA2RGB, 3, streams[i]);
        // FIXME
        //cv::cuda::remap(inputs[i], warped_imgs_scale[i], map1s[i], map2s[i], 
                        //cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), streams[i]);
    }

    for(int i = 0 ; i < nonoverlay_num ; i += 1)
        streams[i].waitForCompletion();
    timer.tick("Uploading and remapping and resizing images");

    std::vector<GpuMat> partial_warped_imgs(warped_imgs.begin(), warped_imgs.begin() + nonoverlay_num);
    std::vector<GpuMat> partial_masks(masks.begin(), masks.begin() + nonoverlay_num);

    if(this->compensator) {
        std::vector<GpuMat> partial_warped_imgs_scale(warped_imgs_scale.begin(),
                                                      warped_imgs_scale.begin() + nonoverlay_num);
        compensator->feed(partial_warped_imgs_scale);
        timer.tick("Compensator");

        compensator->apply(partial_warped_imgs, partial_masks);
        timer.tick("Compensator apply");
    }

    if(this->blender) {
        blender->blend(partial_warped_imgs, result);
        timer.tick("Blender blend");
    } else {
        for(int i = 0 ; i < nonoverlay_num ; i += 1)
            warped_imgs[i].copyTo(result, masks[i], streams[inputs.size()]);
        timer.tick("No blend copy");
    }

    for(int i = nonoverlay_num ; i < inputs.size() ; i += 1) {
        streams[i].waitForCompletion();
        warped_imgs_scale[i].copyTo(result, masks[i], streams[inputs.size()]);
    }
    CV_Assert(result.type() == CV_8UC3);

    if(this->stitch_size != this->scaled_output_size)
        cv::cuda::resize(result, result_scaled, this->scaled_output_size);
    else
        result_scaled = result;

    cv::cuda::cvtRGB24toUYVY422(result_scaled, output, streams[inputs.size()]);
    cv::cuda::GpuMat output_c4 = output.reshape(4);
    cv::cuda::swapChannels(output_c4, swap_orders, streams[inputs.size()]);

    streams[inputs.size()].waitForCompletion();
    timer.tick("Convert output");

    CV_Assert(output.type() == CV_8UC2);

#else
    assert(false);
#endif
}
