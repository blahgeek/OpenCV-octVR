/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-14
*/

#include <iostream>
#include <algorithm>
#include <utility>
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

#ifdef WITH_OCTVR_LOGO
#include "./logo_png.h"
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

    this->nonoverlay_num = mt.inputs.size();

    this->stitch_size = mt.out_size;
    this->scaled_output_size = scale_output.area() == 0 ? mt.out_size : scale_output;

#ifdef WITH_OCTVR_LOGO
    std::vector<unsigned char> logo_data_(OCTVR_LOGO_DATA, OCTVR_LOGO_DATA + OCTVR_LOGO_DATA_LEN);
    cv::Mat logo_png = cv::imdecode(logo_data_, -1);
    CV_Assert(logo_png.type() == CV_8UC4);

    std::vector<cv::Mat> logo_channels(4);
    cv::Mat logo_data_mat, logo_mask_mat;

    cv::split(logo_png, logo_channels);
    logo_mask_mat = logo_channels[3];
    logo_channels.pop_back();

    std::swap(logo_channels[0], logo_channels[2]);
    cv::merge(logo_channels, logo_data_mat);

    GpuMat logo_data_tmp, logo_mask_tmp;
    logo_data_tmp.upload(logo_data_mat);
    logo_mask_tmp.upload(logo_mask_mat);

    cv::cuda::resize(logo_data_tmp, this->logo_data, this->stitch_size);
    cv::cuda::resize(logo_mask_tmp, this->logo_mask, this->stitch_size);
    CV_Assert(this->logo_data.type() == CV_8UC3 && this->logo_mask.type() == CV_8U);
    
    timer.tick("Prepare logo");
#endif

    if(this->nonoverlay_num == 1) {
        std::cerr << "Disable blend and gain compensator since input count = 1" << std::endl;
        enable_gain_compensator = 0;
        blend = 0;
    }

    std::vector<GpuMat> scaled_masks;

    this->map1s.resize(mt.inputs.size() + mt.overlay_inputs.size());
    this->map2s.resize(mt.inputs.size() + mt.overlay_inputs.size());
    this->masks.resize(mt.inputs.size() + mt.overlay_inputs.size());

    this->seam_masks.resize(mt.inputs.size());
    this->rois.resize(mt.inputs.size() + mt.overlay_inputs.size());

    this->working_scale = std::min(1.0, sqrt(WORKING_MEGAPIX * 1e6 / stitch_size.area()));
    for(auto & in: mt.inputs)
        this->working_scaled_rois.emplace_back(in.roi.x * working_scale, 
                                               in.roi.y * working_scale,
                                               in.roi.width * working_scale,
                                               in.roi.height * working_scale);
    scaled_masks.resize(mt.inputs.size());

    for(int i = 0 ; i < nonoverlay_num ; i += 1) {
        map1s[i].upload(mt.inputs[i].map1);
        map2s[i].upload(mt.inputs[i].map2);
        masks[i].upload(mt.inputs[i].mask);
        seam_masks[i].upload(mt.seam_masks[i]);
        rois[i] = mt.inputs[i].roi;
        if(enable_gain_compensator)
            cv::cuda::resize(masks[i], scaled_masks[i], working_scaled_rois[i].size());
    }
    for(int i = 0 ; i < mt.overlay_inputs.size() ; i += 1) {
        map1s[i + nonoverlay_num].upload(mt.overlay_inputs[i].map1);
        map2s[i + nonoverlay_num].upload(mt.overlay_inputs[i].map2);
        masks[i + nonoverlay_num].upload(mt.overlay_inputs[i].mask);
        rois[i + nonoverlay_num] = mt.overlay_inputs[i].roi;
    }

    timer.tick("Uploading mats");

    this->streams.resize(mt.inputs.size() + mt.overlay_inputs.size() + 1);
    this->rgb_inputs.resize(mt.inputs.size() + mt.overlay_inputs.size());
    this->rgba_inputs.resize(mt.inputs.size() + mt.overlay_inputs.size());
    this->warped_imgs.resize(mt.inputs.size() + mt.overlay_inputs.size());

    this->warped_imgs_scale.resize(mt.inputs.size() + mt.overlay_inputs.size());
    this->warped_imgs_rgb.resize(mt.inputs.size() + mt.overlay_inputs.size());
    for(int i = 0 ; i < mt.inputs.size() ; i += 1) {
        rgb_inputs[i].create(in_sizes[i], CV_8UC3);
        rgba_inputs[i].create(in_sizes[i], CV_8UC4);
        warped_imgs[i].create(rois[i].size(), CV_8UC4);
        if(enable_gain_compensator)
            cv::cuda::resize(warped_imgs[i], warped_imgs_scale[i],
                             working_scaled_rois[i].size(), 0, 0, cv::INTER_NEAREST);
        if(blend == 0)
            warped_imgs_rgb[i].create(rois[i].size(), CV_8UC3);
    }

    for(int i = mt.inputs.size() ; i < mt.inputs.size() + mt.overlay_inputs.size() ; i += 1) {
        rgb_inputs[i].create(in_sizes[i], CV_8UC3);
        rgba_inputs[i].create(in_sizes[i], CV_8UC4);
        warped_imgs[i].create(rois[i].size(), CV_8UC4);
        warped_imgs_rgb[i].create(rois[i].size(), CV_8UC3);
    }

    this->result.create(stitch_size, CV_8UC3);
    if(this->stitch_size != this->scaled_output_size)
        this->result_scaled.create(this->scaled_output_size, CV_8UC3);
    this->result.setTo(0);

    timer.tick("Allocating internal mats");

    if(blend > 0) {
        int blend_bands = int(ceil(log(blend)/log(2.)) - 1.);
        std::cerr << "Using MultiBandBlender with band number = " << blend_bands << std::endl;
        this->blender.reset(new cv::detail::MultiBandGPUBlender(seam_masks, 
                                                                std::vector<cv::Rect>(rois.begin(), rois.begin() + nonoverlay_num),
                                                                blend_bands));
    } else if(blend < 0) {
        //float sharpness = 1.0 / float(-blend);
        std::cerr << "Using FeatherBlender with border = " << -blend << std::endl;
        this->blender.reset(new cv::detail::FeatherGPUBlender(std::vector<GpuMat>(masks.begin(), masks.begin() + nonoverlay_num), 
                                                              std::vector<cv::Rect>(rois.begin(), rois.begin() + nonoverlay_num),
                                                              -blend));
    } else {
        std::cerr << "Do not use blender" << std::endl;
        this->blender = nullptr;
    }

    timer.tick("Blender initialize");

    if(enable_gain_compensator)
        this->compensator.reset(new cv::detail::GainCompensatorGPU(scaled_masks, working_scaled_rois));
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
                    GpuMat & output, GpuMat & preview_output,
                    bool mix_input_channels) {
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

    const int swap_orders[] = {1, 0, 3, 2};
    
    for(int i = 0 ; i < nonoverlay_num ; i += 1) {
        if(mix_input_channels) {
            cv::cuda::GpuMat input_c4 = inputs[i].reshape(4);
            cv::cuda::swapChannels(input_c4, swap_orders, streams[i]);
        }
        cv::cuda::cvtYUYV422toRGB24(inputs[i], rgb_inputs[i], streams[i]);
        cv::cuda::cvtColor(rgb_inputs[i], rgba_inputs[i], cv::COLOR_RGB2RGBA, 4, streams[i]);
        cv::cuda::fastRemap(rgba_inputs[i], warped_imgs[i], map1s[i], map2s[i], true, streams[i]);
        if(this->compensator)
            cv::cuda::resize(warped_imgs[i], warped_imgs_scale[i],
                             working_scaled_rois[i].size(), 0, 0,
                             cv::INTER_NEAREST, streams[i]);
    }

    for(int i = nonoverlay_num ; i < inputs.size() ; i += 1) {
        if(mix_input_channels) {
            cv::cuda::GpuMat input_c4 = inputs[i].reshape(4);
            cv::cuda::swapChannels(input_c4, swap_orders, streams[i]);
        }
        cv::cuda::cvtYUYV422toRGB24(inputs[i], rgb_inputs[i], streams[i]);
        cv::cuda::cvtColor(rgb_inputs[i], rgba_inputs[i], cv::COLOR_RGB2RGBA, 4, streams[i]);
        cv::cuda::fastRemap(rgba_inputs[i], warped_imgs[i], map1s[i], map2s[i], false, streams[i]);
        cv::cuda::cvtColor(warped_imgs[i], warped_imgs_rgb[i], cv::COLOR_RGBA2RGB, 3, streams[i]);
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

    cv::cuda::Stream & stream_final = streams[inputs.size()];

    if(this->blender) {
        blender->blend(partial_warped_imgs, result);
        timer.tick("Blender blend");
    } else {
        for(int i = 0 ; i < nonoverlay_num ; i += 1) {
            cv::cuda::cvtColor(warped_imgs[i], warped_imgs_rgb[i], cv::COLOR_RGBA2RGB, 3, stream_final);
            warped_imgs_rgb[i].copyTo(result(rois[i]), masks[i], stream_final);
        }
        timer.tick("No blend copy");
    }

    for(int i = nonoverlay_num ; i < inputs.size() ; i += 1) {
        streams[i].waitForCompletion();
        warped_imgs_rgb[i].copyTo(result(rois[i]), masks[i], stream_final);
    }
    CV_Assert(result.type() == CV_8UC3);

#ifdef WITH_OCTVR_LOGO
    this->logo_data.copyTo(result, this->logo_mask, stream_final);
#endif

    if(this->stitch_size != this->scaled_output_size)
        cv::cuda::resize(result, result_scaled, this->scaled_output_size,
                         0, 0, cv::INTER_LINEAR, stream_final);
    else
        result_scaled = result;

    cv::cuda::cvtRGB24toYUYV422(result_scaled, output, stream_final);
    cv::cuda::GpuMat output_c4 = output.reshape(4);
    cv::cuda::swapChannels(output_c4, swap_orders, stream_final);

    if(!preview_output.empty()) {
        CV_Assert(preview_output.type() == CV_8UC3);
        cv::cuda::resize(result, preview_output, preview_output.size(), 
                         0, 0, cv::INTER_LINEAR, stream_final);
    }

    stream_final.waitForCompletion();
    timer.tick("Convert output");

    CV_Assert(output.type() == CV_8UC2);

#else
    assert(false);
#endif
}
