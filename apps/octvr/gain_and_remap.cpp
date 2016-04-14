/* 
* @Author: BlahGeek
* @Date:   2016-03-15
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-14
*/

#include <iostream>
#include <fstream>

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

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
#define BLUR_BLOCK 16.0

int main(int argc, char const *argv[]) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " stitch.dat img0.png ... rmap0.dat ... defish0.dat ... [defish0.dat ...]" << std::endl;
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
    CV_Assert((argc - 2) % stitch_template.inputs.size() == 0);

    argv += 2;
    argc -= 2;

    // load sources
    std::vector<cv::UMat> src_images;
    for(size_t i = 0 ; i < stitch_template.inputs.size() ; i += 1)
        src_images.push_back(load_image(argv[i]));

    argv += stitch_template.inputs.size();
    argc -= stitch_template.inputs.size();

    // remap
    std::vector<cv::UMat> stitch_remap_images(stitch_template.inputs.size());
    for(size_t i = 0 ; i < stitch_template.inputs.size() ; i += 1)
        cv::remap(src_images[i], stitch_remap_images[i], 
                  stitch_template.inputs[i].map1 * src_images[i].cols,
                  stitch_template.inputs[i].map2 * src_images[i].rows, 
                  cv::INTER_LINEAR);

    // scale
    double working_scale = sqrt(WORKING_MEGAPIX * 1e6 / stitch_template.out_size.area());
    if(working_scale > 1.0)
        working_scale = 1.0;
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
    auto exposure_compensator = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
    exposure_compensator->feed(scaled_corners, scaled_stitch_remap_images, scaled_masks);
    auto gain_maps = dynamic_cast<cv::detail::BlocksGainCompensator *>(exposure_compensator.get())->getGainMaps();

    // apply gains
    for(size_t i = 0 ; i < stitch_template.inputs.size() ; i += 1) {
        auto r_template = load_template(argv[i]);
        CV_Assert(r_template.inputs.size() == 1);
        CV_Assert(r_template.out_size == src_images[i].size());

        cv::UMat full_gain_map(stitch_template.out_size, CV_32F);
        full_gain_map.setTo(0);
        cv::resize(gain_maps[i], full_gain_map(stitch_template.inputs[i].roi), 
                   stitch_template.inputs[i].roi.size(), 0, 0, cv::INTER_LINEAR);
        cv::UMat orig_gain_map;
        cv::remap(full_gain_map, orig_gain_map,
                  r_template.inputs[0].map1 * stitch_template.out_size.width,
                  r_template.inputs[0].map2 * stitch_template.out_size.height,
                  cv::INTER_LINEAR, cv::BORDER_WRAP);

        int blur_size = ((int(BLUR_BLOCK / working_scale) >> 1) << 1) + 1;
        // cv::GaussianBlur(orig_gain_map, orig_gain_map, cv::Size(blur_size, blur_size), 0);
        cv::blur(orig_gain_map, orig_gain_map, cv::Size(blur_size, blur_size));

#if 0
        cv::UMat tmp_m;
        orig_gain_map.convertTo(tmp_m, CV_8U, 255.0);
        char tmp[128];
        snprintf(tmp, 128, "debug/gain_%lu.png", i);
        cv::imwrite(tmp, tmp_m);

        full_gain_map.convertTo(tmp_m, CV_8U, 255.0);
        snprintf(tmp, 128, "debug/full_%lu.png", i);
        cv::imwrite(tmp, tmp_m);
#endif

        cv::Mat_<float> gain = orig_gain_map.getMat(cv::ACCESS_READ);

        cv::Mat image = src_images[i].getMat(cv::ACCESS_RW);
        for (int y = 0; y < image.rows; ++y) {
            const float* gain_row = gain.ptr<float>(y);
            cv::Point3_<uchar>* row = image.ptr<cv::Point3_<uchar> >(y);
            for (int x = 0; x < image.cols; ++x) {
                row[x].x = cv::saturate_cast<uchar>(row[x].x * gain_row[x]);
                row[x].y = cv::saturate_cast<uchar>(row[x].y * gain_row[x]);
                row[x].z = cv::saturate_cast<uchar>(row[x].z * gain_row[x]);
            }
        }
    }

    argv += stitch_template.inputs.size();
    argc -= stitch_template.inputs.size();

    char const ** orig_filenames = argv - 2 * stitch_template.inputs.size();

    // output
    if(argc > 0) {
        int defish_count = 0;
        while(argc > 0) {
            defish_count += 1;
            for(size_t i = 0 ; i < stitch_template.inputs.size() ; i += 1) {
                auto output_template = load_template(argv[i]);
                CV_Assert(output_template.inputs.size() == 1);
                cv::UMat remapped;
                cv::remap(src_images[i], remapped,
                          output_template.inputs[0].map1 * src_images[i].cols,
                          output_template.inputs[0].map2 * src_images[i].rows,
                          cv::INTER_LINEAR);
                char suffix[128];
                snprintf(suffix, 128, ".defish-%d.png", defish_count);
                save_image(orig_filenames[i], suffix, remapped);
            }

            argc -= stitch_template.inputs.size();
            argv += stitch_template.inputs.size();
        }
    }
    else
        for(size_t i = 0 ; i < stitch_template.inputs.size() ; i += 1)
            save_image(orig_filenames[i], ".gain.png", src_images[i]);

    /* code */
    return 0;
}
