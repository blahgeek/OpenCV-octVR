/* 
* @Author: BlahGeek
* @Date:   2016-03-15
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-15
*/

#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"

#include "octvr.hpp"

using namespace vr;

#define WORKING_MEGAPIX 0.1

int main(int argc, char const *argv[]) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " stitch.dat img0.bmp ... map0.dat ..." << std::endl;
        return 1;
    }

    // helper functions
    auto load_template = [](char const * filename) {
        std::cerr << "Loading template " << filename << std::endl;
        std::ifstream template_file(filename, std::ios::binary);
        return MapperTemplate(template_file);
    };
    auto load_image = [](char const * filename) {
        std::cerr << "Reading image " << filename << std::endl;
        cv::UMat u_img;
        cv::Mat img = cv::imread(filename, 1); // 3-channel color
        img.copyTo(u_img);
        return u_img;
    };
    auto save_image = [](char const * filename, char const * suffix, cv::UMat img) {
        std::string name = std::string(filename) + std::string(suffix);
        std::cerr << "Writing image " << name << std::endl;
        cv::imwrite(name, img);
    };

    // load stitching template
    MapperTemplate stitch_template = load_template(argv[1]);
    std::cerr << stitch_template.inputs.size() << " images found" << std::endl;
    assert(argc - 2 == stitch_template.inputs.size() / 2);

    argv += 2;

    // load sources
    std::vector<cv::UMat> src_images;
    for(size_t i = 0 ; i < stitch_template.inputs.size() ; i += 1)
        src_images.push_back(load_image(argv[i]));

    argv += stitch_template.inputs.size();

    // remap
    std::vector<cv::UMat> stitch_remap_images(stitch_template.inputs.size());
    for(size_t i = 0 ; i < stitch_template.inputs.size() ; i += 1)
        cv::remap(src_images[i], stitch_remap_images[i], 
                  stitch_template.inputs[i].map1 * src_images[i].cols,
                  stitch_template.inputs[i].map2 * src_images[i].rows, 
                  cv::INTER_LINEAR);

    // scale
    double working_scale = std::min(1.0, sqrt(WORKING_MEGAPIX * 1e6 / stitch_template.out_size.area()));
    std::vector<cv::Rect> scaled_rois;
    for(auto & in: stitch_template.inputs)
        scaled_rois.emplace_back(in.roi.x * working_scale, 
                                 in.roi.y * working_scale,
                                 in.roi.width * working_scale,
                                 in.roi.height * working_scale);

    std::vector<cv::UMat> scaled_stitch_remap_images(stitch_template.inputs.size());
    std::vector<cv::UMat> scaled_masks(stitch_template.inputs.size());
    std::vector<cv::Point> scaled_corners;
    for(size_t i = 0 ; i < stitch_template.inputs.size() ; i += 1) {
        cv::resize(stitch_remap_images[i], scaled_stitch_remap_images[i], scaled_rois[i].size());
        cv::resize(stitch_template.inputs[i].mask, scaled_masks[i], scaled_rois[i].size());
        scaled_corners.push_back(scaled_rois[i].tl());
    }

    // exposure!
    auto exposure_compensator = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN);
    exposure_compensator->feed(scaled_corners, scaled_stitch_remap_images, scaled_masks);
    auto gains = dynamic_cast<cv::detail::GainCompensator *>(exposure_compensator.get())->gains();

    // output
    for(size_t i = 0 ; i < stitch_template.inputs.size() ; i += 1) {
        auto output_template = load_template(argv[i]);
        assert(output_template.inputs.size() == 1);
        cv::UMat remapped;
        cv::remap(src_images[i], remapped,
                  output_template.inputs[0].map1 * src_images[i].cols,
                  output_template.inputs[0].map2 * src_images[i].rows,
                  cv::INTER_LINEAR);
        exposure_compensator->apply(i, scaled_corners[i], remapped, cv::UMat());
        save_image(argv[i - stitch_template.inputs.size()], ".defish.bmp", remapped);
    }

    /* code */
    return 0;
}
