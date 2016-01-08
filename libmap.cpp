/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-08
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

// TODO Set by options
#define WORKING_MEGAPIX 0.1

using namespace vr;

Mapper::Mapper(const MapperTemplate & mt, 
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

void Mapper::stitch(const std::vector<UMat> & inputs, UMat & output) {
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

// Mapper::Mapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes, int blend) {
// #ifdef WITH_DONGLE_LICENSE
//     lic_runtime_init(&(this->lic_t), 601);
//     this->lic_cnt = 0;
// #endif

// #ifdef HAVE_CUDA
//     Timer timer("Mapper constructor");

//     std::vector<GpuMat> scaled_masks;

//     this->nonoverlay_num = mt.inputs.size();

//     this->out_size = mt.out_size;
//     this->map1s.resize(mt.inputs.size() + mt.overlay_inputs.size());
//     this->map2s.resize(mt.inputs.size() + mt.overlay_inputs.size());
//     this->masks.resize(mt.inputs.size() + mt.overlay_inputs.size());

//     this->seam_masks.resize(mt.inputs.size());
//     scaled_masks.resize(mt.inputs.size());

//     this->working_scale = std::min(1.0, sqrt(WORKING_MEGAPIX * 1e6 / out_size.area()));

//     for(int i = 0 ; i < nonoverlay_num ; i += 1) {
//         map1s[i].upload(mt.inputs[i].map1);
//         map2s[i].upload(mt.inputs[i].map2);
//         masks[i].upload(mt.inputs[i].mask);
//         seam_masks[i].upload(mt.seam_masks[i]);
//         cv::cuda::resize(masks[i], scaled_masks[i],
//                          cv::Size(), working_scale, working_scale);
//     }
//     for(int i = 0 ; i < mt.overlay_inputs.size() ; i += 1) {
//         map1s[i + nonoverlay_num].upload(mt.overlay_inputs[i].map1);
//         map2s[i + nonoverlay_num].upload(mt.overlay_inputs[i].map2);
//         masks[i + nonoverlay_num].upload(mt.overlay_inputs[i].mask);
//     }

//     timer.tick("Uploading mats");

//     this->streams.resize(mt.inputs.size() + mt.overlay_inputs.size() + 1);
//     this->gpu_inputs.resize(mt.inputs.size() + mt.overlay_inputs.size());
//     this->warped_imgs.resize(mt.inputs.size() + mt.overlay_inputs.size());

//     this->warped_imgs_scale.resize(mt.inputs.size() + mt.overlay_inputs.size());
//     for(int i = 0 ; i < mt.inputs.size() ; i += 1) {
//         gpu_inputs[i].create(in_sizes[i], CV_8UC4);
//         warped_imgs[i].create(out_size, CV_8UC4);
//         cv::cuda::resize(warped_imgs[i], warped_imgs_scale[i],
//                          cv::Size(), working_scale, working_scale,
//                          cv::INTER_NEAREST);
//     }

//     for(int i = mt.inputs.size() ; i < mt.inputs.size() + mt.overlay_inputs.size() ; i += 1) {
//         gpu_inputs[i].create(in_sizes[i], CV_8UC4);
//         warped_imgs[i].create(out_size, CV_8UC4);
//         warped_imgs_scale[i].create(out_size, CV_8UC3);  // NOTICE
//     }

//     this->result.create(out_size, CV_8UC3);

//     if(blend == 0)
//         return;

//     timer.tick("Allocating internal mats");

//     if(blend > 0) {
//         int blend_bands = int(ceil(log(blend)/log(2.)) - 1.);
//         std::cerr << "Using MultiBandBlender with band number = " << blend_bands << std::endl;
//         this->blender = cv::makePtr<cv::detail::MultiBandGPUBlender>(seam_masks, blend_bands);
//     } else {
//         //float sharpness = 1.0 / float(-blend);
//         std::cerr << "Using FeatherBlender with border = " << -blend << std::endl;
//         this->blender = cv::makePtr<cv::detail::FeatherGPUBlender>(std::vector<GpuMat>(masks.begin(), masks.begin() + nonoverlay_num), 
//                                                                    -blend);
//     }

//     timer.tick("Blender initialize");

//     this->compensator = cv::makePtr<cv::detail::GainCompensatorGPU>(scaled_masks);
//     timer.tick("Gain Compensator initialize");
// #else
//     assert(false);
// #endif

// }

// void Mapper::stitch(const std::vector<GpuMat> & inputs,
//                     GpuMat & output) {
// #ifdef WITH_DONGLE_LICENSE
//     this->lic_cnt += 1;
//     if (this->lic_cnt % 3000 == 0) {
//         lic_runtime_check(&(this->lic_t));
//     }
// #endif

// #ifdef HAVE_CUDA
//     Timer timer("Stitch");

//     assert(inputs.size() == masks.size());
//     for(int i = 0 ; i < inputs.size() ; i += 1)
//         assert(inputs[i].type() == CV_8UC3); // RGB
//     assert(output.type() == CV_8UC3 && output.size() == this->out_size);
    
//     std::vector<cv::Point2i> corners;

//     for(int i = 0 ; i < nonoverlay_num ; i += 1) {
//         cv::cuda::cvtColor(inputs[i], gpu_inputs[i], cv::COLOR_RGB2RGBA, 4, streams[i]);
//         cv::cuda::fastRemap(gpu_inputs[i], warped_imgs[i], map1s[i], map2s[i], true, streams[i]);
//         cv::cuda::resize(warped_imgs[i], warped_imgs_scale[i],
//                          cv::Size(), working_scale, working_scale,
//                          cv::INTER_NEAREST, streams[i]);
//         corners.emplace_back(0, 0);
//     }

//     for(int i = nonoverlay_num ; i < inputs.size() ; i += 1) {
//         cv::cuda::cvtColor(inputs[i], gpu_inputs[i], cv::COLOR_RGB2RGBA, 4, streams[i]);
//         cv::cuda::fastRemap(gpu_inputs[i], warped_imgs[i], map1s[i], map2s[i], false, streams[i]);
//         cv::cuda::cvtColor(warped_imgs[i], warped_imgs_scale[i], cv::COLOR_RGBA2RGB, 3, streams[i]);
//         // FIXME
//         //cv::cuda::remap(inputs[i], warped_imgs_scale[i], map1s[i], map2s[i], 
//                         //cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), streams[i]);
//     }

//     for(int i = 0 ; i < nonoverlay_num ; i += 1)
//         streams[i].waitForCompletion();
//     timer.tick("Uploading and remapping and resizing images");

//     std::vector<GpuMat> partial_warped_imgs_scale(warped_imgs_scale.begin(),
//                                                   warped_imgs_scale.begin() + nonoverlay_num);
//     compensator->feed(partial_warped_imgs_scale);
//     timer.tick("Compensator");

//     std::vector<GpuMat> partial_warped_imgs(warped_imgs.begin(), warped_imgs.begin() + nonoverlay_num);
//     std::vector<GpuMat> partial_masks(masks.begin(), masks.begin() + nonoverlay_num);

//     compensator->apply(partial_warped_imgs, partial_masks);
//     timer.tick("Compensator apply");

//     blender->blend(partial_warped_imgs, output);
//     timer.tick("Blender blend");

//     for(int i = nonoverlay_num ; i < inputs.size() ; i += 1) {
//         streams[i].waitForCompletion();
//         warped_imgs_scale[i].copyTo(output, masks[i], streams[inputs.size()]);
//     }

//     streams[inputs.size()].waitForCompletion();

//     assert(output.type() == CV_8UC3);

// #else
//     assert(false);
// #endif
// }

// void Mapper::remap(const std::vector<GpuMat> & inputs,
//                        GpuMat & output) {
//     Timer timer("Remap");

// #ifdef HAVE_CUDA
//     assert(inputs.size() == masks.size());
//     for(int i = 0 ; i < inputs.size() ; i += 1)
//         assert(inputs[i].type() == CV_8UC3); // RGB

//     assert(output.type() == CV_8UC3);
//     assert(output.size() == this->out_size);
    
//     for(int i = 0 ; i < inputs.size() ; i += 1) {
//         cv::cuda::cvtColor(inputs[i], gpu_inputs[i], cv::COLOR_RGB2RGBA, 4, streams[i]);
//         cv::cuda::fastRemap(gpu_inputs[i], warped_imgs[0], map1s[i], map2s[i], false, streams[i]);
//     }

//     for(auto & s: streams)
//         s.waitForCompletion();
//     timer.tick("Remapping images");

//     cv::cuda::cvtColor(warped_imgs[0], output, cv::COLOR_RGBA2RGB, 3, streams[0]);
//     streams[0].waitForCompletion();

//     timer.tick("Convert");

// #else 
//     assert(false);
// #endif

// }
