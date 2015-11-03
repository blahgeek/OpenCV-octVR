/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-27
*/

#include <iostream>
#include "./libmap_impl.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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

using namespace vr;

MultiMapper * MultiMapper::New(const std::string & to, const json & to_opts,
                               int out_width, int out_height) {
    return new MultiMapperImpl(to, to_opts, out_width, out_height);
}

MultiMapperImpl::MultiMapperImpl(const std::string & to, const json & to_opts,
                                 int out_width, int out_height) {
    this->out_camera = Camera::New(to, to_opts);
    if(!this->out_camera)
        throw std::string("Invalid output camera type");

    if(out_height <= 0 && out_width <= 0)
        throw std::string("Output width/height invalid");
    double output_aspect_ratio = this->out_camera->get_aspect_ratio();
    if(out_height <= 0)
        out_height = int(double(out_width) / output_aspect_ratio);
    if(out_width <= 0)
        out_width = int(double(out_height) * output_aspect_ratio);
    std::cerr << "Output size: " << out_width << "x" << out_height << std::endl;
    this->out_size = cv::Size(out_width, out_height);

    std::vector<cv::Point2d> tmp;
    for(int j = 0 ; j < out_height ; j += 1)
        for(int i = 0 ; i < out_width ; i += 1)
            tmp.push_back(cv::Point2d(double(i) / out_width,
                                      double(j) / out_height));
    this->output_map_points = this->out_camera->image_to_obj(tmp);
}

void MultiMapperImpl::add_input(const std::string & from, const json & from_opts,
                                int in_width, int in_height) {
    this->in_sizes.push_back(cv::Size(in_width, in_height));
    auto _cam = Camera::New(from, from_opts);
    if(!_cam)
        throw std::string("Invalid input camera type");
    this->in_cameras.push_back(std::move(_cam));

    auto tmp = this->in_cameras.back()->obj_to_image(this->output_map_points);
    cv::Mat orig_map1(out_size, CV_32FC1), orig_map2(out_size, CV_32FC1);
    cv::Mat mask(out_size, CV_8U);
    for(int h = 0 ; h < out_size.height ; h += 1) {
        unsigned char * mask_row = mask.ptr(h);
        float * map1_row = orig_map1.ptr<float>(h);
        float * map2_row = orig_map2.ptr<float>(h);

        for(int w = 0 ; w < out_size.width ; w += 1) {
            auto index = w + out_size.width * h;
            float x = tmp[index].x * in_width;
            float y = tmp[index].y * in_height;
            if(isnan(x) || isnan(y) ||
               x < 0 || x >= in_width || y < 0 || y >= in_height) {
                mask_row[w] = 0;
                map1_row[w] = map2_row[w] = NAN;
            }
            else {
                mask_row[w] = 255;
                map1_row[w] = x;
                map2_row[w] = y;
            }
        }
    }
    cv::Mat map1(out_size, CV_16SC2), map2(out_size, CV_16UC1);
    cv::convertMaps(orig_map1, orig_map2, map1, map2, CV_16SC2);

    this->map1s.push_back(std::move(map1));
    this->map2s.push_back(std::move(map2));
    this->masks.push_back(std::move(mask));
}

void MultiMapperImpl::get_output(const std::vector<cv::Mat> & inputs, cv::Mat & output) {
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        assert(inputs[i].type() == CV_8UC3);
        assert(inputs[i].size() == in_sizes[i]);
    }
    assert(output.type() == CV_8UC3 && output.size() == this->out_size);
    
    std::vector<cv::Point2i> corners;
    std::vector<cv::Size> sizes;
    std::vector<cv::Mat> warped_imgs_float, warped_imgs_uchar, warped_imgs_short;
    std::vector<cv::Mat> masks_clone;
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        cv::Mat warped_img_uchar;
        cv::Mat warped_img_float;
        cv::Mat warped_img_short;

        cv::remap(inputs[i], warped_img_uchar, map1s[i], map2s[i], CV_INTER_CUBIC);
        warped_img_uchar.convertTo(warped_img_float, CV_32FC3, 1.0/255);
        warped_img_uchar.convertTo(warped_img_short, CV_16SC3);

        warped_imgs_uchar.push_back(std::move(warped_img_uchar));
        warped_imgs_float.push_back(std::move(warped_img_float));
        warped_imgs_short.push_back(std::move(warped_img_short));

        corners.emplace_back(0, 0);
        sizes.push_back(out_size);
        masks_clone.push_back(this->masks[i].clone());
    }

    // TODO
    // GraphCut would run into a infinity loop if mask is cut by border
    // auto seam_finder = new cv::detail::GraphCutSeamFinder(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
    auto seam_finder = new cv::detail::DpSeamFinder(cv::detail::DpSeamFinder::COLOR);
    // cv::Ptr<cv::detail::SeamFinder> seam_finder = new cv::detail::VoronoiSeamFinder();
    seam_finder->find(warped_imgs_float, corners, masks_clone);

    auto blender = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, false);
    // auto blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, false);
    // TODO set band number
    dynamic_cast<cv::detail::MultiBandBlender *>(static_cast<cv::detail::Blender *>(blender))->setNumBands(3);
    blender->prepare(corners, sizes);
    for(int i = 0 ; i < inputs.size() ; i += 1)
        blender->feed(warped_imgs_short[i], masks[i], corners[i]);

    cv::Mat result, result_mask;
    blender->blend(result, result_mask);

    result.convertTo(output, CV_8UC3);
}
