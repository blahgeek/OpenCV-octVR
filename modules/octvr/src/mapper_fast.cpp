/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-01
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

using namespace vr;

FastMapper::FastMapper(const MapperTemplate & mt, 
                       std::vector<cv::Size> in_sizes) {
    Timer timer("FastMapper constructor");

    CV_Assert(mt.overlay_inputs.size() == 0);

    this->in_sizes = in_sizes;
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

    for(int i = 0 ; i < mt.inputs.size() ; i += 1) {
        auto & in = mt.inputs[i];

        cv::Mat r_map1, r_map2;
        cv::resize(in.map1, r_map1, cv::Size(in.map1.cols / 2, in.map1.rows / 2));
        cv::resize(in.map2, r_map2, cv::Size(in.map2.cols / 2, in.map2.rows / 2));

        cv::UMat half_map1, half_map2;
        cv::convertMaps(r_map1 * (in_sizes[i].width / 2), 
                        r_map2 * (in_sizes[i].height / 2),
                        half_map1, half_map2, CV_16SC2);
        this->half_map1s.push_back(half_map1);
        this->half_map2s.push_back(half_map2);

        cv::UMat mask;
        cv::resize(in.mask, mask, cv::Size(in.mask.cols / 2, in.mask.rows / 2));
        this->half_masks.push_back(mask);
    }

    timer.tick("Copying maps and masks");

    std::vector<cv::UMat> weight_maps;
    cv::UMat dst_weight_map(out_size, CV_32F);
    dst_weight_map.setTo(1e-5f);
    for(int i = 0 ; i < mt.inputs.size() ; i += 1) {
        cv::UMat weight_map, tmp;
        cv::distanceTransform(masks[i], weight_map, cv::DIST_L2, 3);
        cv::subtract(weight_map, 5, tmp);
        cv::threshold(tmp, weight_map, 0.f, 0.f, cv::THRESH_TOZERO);
        cv::add(weight_map, dst_weight_map, dst_weight_map);
        weight_maps.push_back(weight_map);
    }

    this->feather_masks.resize(weight_maps.size());
    this->half_feather_masks.resize(weight_maps.size());
    for(int i = 0 ; i < mt.inputs.size() ; i += 1) {
        cv::divide(weight_maps[i], dst_weight_map, weight_maps[i]);
        weight_maps[i].convertTo(feather_masks[i], CV_8U, 255.0);
        cv::resize(feather_masks[i], half_feather_masks[i],
                   cv::Size(feather_masks[i].cols / 2, feather_masks[i].rows / 2));
    }

    timer.tick("Prepare feather masks");

    output_s_c0.create(this->out_size, CV_16U);
    output_s_c1c2.emplace_back(this->out_size.height / 2,
                               this->out_size.width / 2,
                               CV_16U);
    output_s_c1c2.emplace_back(this->out_size.height / 2,
                               this->out_size.width / 2,
                               CV_16U);
    input_c1c2.emplace_back(in_sizes[0].height / 2, in_sizes[0].width / 2, CV_8U);
    input_c1c2.emplace_back(in_sizes[0].height / 2, in_sizes[0].width / 2, CV_8U);

    output_c1c2_merge.create(this->out_size.height / 2, this->out_size.width / 2, CV_16UC2);
}

void FastMapper::stitch(const std::vector<UMat> & inputs, UMat & output) {
    #if 0
    Timer timer("stitch");

    output.create(this->out_size, CV_8UC3);
    output.setTo(0);

    std::vector<UMat> output_channels(3);
    for(int k = 0 ; k < 3 ; k += 1)
        output_channels[k].create(this->out_size, CV_32F);

    for(int i = 0 ; i < this->masks.size() ; i += 1) {
        CV_Assert(inputs[i].type() == CV_8UC3);

        std::vector<UMat> input_channels(3);

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
    #endif

    throw "not supported yet";

}

void FastMapper::stitch_nv12(const std::vector<cv::UMat> & inputs, cv::UMat & output) {
    Timer timer("stitch");

    for(int i = 0 ; i < inputs.size() ; i += 1) {
        CV_Assert(inputs[i].rows == in_sizes[i].height + in_sizes[i].height / 2);
        CV_Assert(inputs[i].cols == in_sizes[i].width);
        CV_Assert(inputs[i].type() == CV_8U);
    }

    output.create(this->out_size.height + this->out_size.height / 2,
                  this->out_size.width,
                  CV_8U);

    output_s_c0.setTo(0);
    output_s_c1c2[0].setTo(0);
    output_s_c1c2[1].setTo(0);
    timer.tick("Prepare output");

    for(int i = 0 ; i < inputs.size() ; i += 1) {
        cv::UMat input_c0 = inputs[i].rowRange(0, in_sizes[i].height);
        cv::UMat input_c1c2_orig = inputs[i].rowRange(in_sizes[i].height, 
                                                      in_sizes[i].height + in_sizes[i].height / 2);
        cv::split(input_c1c2_orig.reshape(2), input_c1c2);
        timer.tick("split input");

        cv::remap_weighted(input_c0, output_s_c0, map1s[i], map2s[i], feather_masks[i], cv::INTER_LINEAR);
        cv::remap_weighted(input_c1c2[0], output_s_c1c2[0], half_map1s[i], half_map2s[i], half_feather_masks[i], cv::INTER_LINEAR);
        cv::remap_weighted(input_c1c2[1], output_s_c1c2[1], half_map1s[i], half_map2s[i], half_feather_masks[i], cv::INTER_LINEAR);
        timer.tick("remap");
    }

    cv::UMat output_c0 = output.rowRange(0, out_size.height);
    output_s_c0.convertTo(output_c0, CV_8U, 1.0/255.0);
    timer.tick("convertTo c0");

    cv::merge(output_s_c1c2, output_c1c2_merge);

    cv::UMat output_c1c2 = output.rowRange(out_size.height, 
                                           out_size.height + out_size.height / 2)
                                 .reshape(2);
    output_c1c2_merge.convertTo(output_c1c2, CV_8U, 1.0/255.0);
    timer.tick("convertTo c1c2");
}
