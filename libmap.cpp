/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-12
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

#include <sys/time.h>

static int64_t gettime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

using namespace vr;

MultiMapper * MultiMapper::New(const std::string & to, const json & to_opts,
                               int out_width, int out_height) {
    return new MultiMapperImpl(to, to_opts, out_width, out_height);
}

MultiMapper * MultiMapper::New(std::ifstream & f) {
    return new MultiMapperImpl(f);
}

MultiMapperImpl::MultiMapperImpl(const std::string & to, const json & to_opts,
                                 int out_width, int out_height) {
    std::unique_ptr<Camera> out_camera = Camera::New(to, to_opts);
    if(!out_camera)
        throw std::string("Invalid output camera type");

    if(out_height <= 0 && out_width <= 0)
        throw std::string("Output width/height invalid");
    double output_aspect_ratio = out_camera->get_aspect_ratio();
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
    this->output_map_points = out_camera->image_to_obj(tmp);
}

void MultiMapperImpl::add_input(const std::string & from, const json & from_opts,
                                int in_width, int in_height) {
    // If this is constructed using dumped data file, add_input is not available
    assert(!this->output_map_points.empty());

    this->in_sizes.push_back(cv::Size(in_width, in_height));
    std::unique_ptr<Camera> cam = Camera::New(from, from_opts);
    if(!cam)
        throw std::string("Invalid input camera type");

    auto tmp = cam->obj_to_image(this->output_map_points);
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
    cv::UMat map1(out_size, CV_16SC2), map2(out_size, CV_16UC1);
    cv::convertMaps(orig_map1, orig_map2, map1, map2, CV_16SC2);

    this->map1s.push_back(map1);
    this->map2s.push_back(map2);
    cv::UMat mask_u;
    mask.copyTo(mask_u);
    this->masks.push_back(mask_u);
}

#ifdef DEBUG_SAVE_MAT

#define SAVE_MAT(N, S, MAT) \
    do { \
        char tmp[64]; \
        snprintf(tmp, 64, S "_%d.jpg", N);\
        imwrite(tmp, MAT.getMat(cv::ACCESS_READ)); \
    } while(false)

#define SAVE_MAT_VEC(S, MATS) \
    do { \
        for(int __i = 0 ; __i < MATS.size() ; __i += 1) \
            SAVE_MAT(__i, S, MATS[__i]); \
    } while(false)

#else

#define SAVE_MAT_VEC(S, MATS)

#endif

#define TIMER(S) \
    do { \
        auto __t = gettime(); \
        std::cerr << S << ": " << (__t - _timer) / 1000.0 << "ms" << std::endl; \
        _timer = __t; \
    } while(false)

void MultiMapperImpl::get_output(const std::vector<cv::UMat> & inputs, cv::UMat & output) {
    auto _timer = gettime();

    // TODO set scale
    auto scale = 0.5;

    for(int i = 0 ; i < inputs.size() ; i += 1) {
        assert(inputs[i].type() == CV_8UC3);
        assert(inputs[i].size() == in_sizes[i]);
    }
    assert(output.type() == CV_8UC3 && output.size() == this->out_size);
    
    std::vector<cv::Point2i> corners;
    std::vector<cv::Size> sizes;
    std::vector<cv::UMat> warped_imgs_uchar(inputs.size());
    
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        cv::remap(inputs[i], warped_imgs_uchar[i], map1s[i], map2s[i], CV_INTER_CUBIC);
        corners.emplace_back(0, 0);
        sizes.push_back(out_size);
    }
    TIMER("Remapping images");

    SAVE_MAT_VEC("warped_img", warped_imgs_uchar);
    SAVE_MAT_VEC("warped_mask", masks);

    std::vector<cv::UMat> warped_imgs_uchar_scale(inputs.size());
    std::vector<cv::UMat> masks_scale(inputs.size());
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        cv::resize(warped_imgs_uchar[i], warped_imgs_uchar_scale[i], cv::Size(), scale, scale);
        cv::resize(this->masks[i], masks_scale[i], cv::Size(), scale, scale);
    }
    TIMER("Scale");

    // TODO GAIN_BLOCKS bugs?
    cv::Ptr<cv::detail::ExposureCompensator> compensator = 
        cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN);
    compensator->feed(corners, warped_imgs_uchar_scale, masks_scale);
    TIMER("Compensator");

    // TODO Optimize this
    // GainCompensator::apply does img *= gain for every image
    // while size of image is large (output size), and only part of it is valid (mask)
    for(int i = 0 ; i < inputs.size() ; i += 1)
        compensator->apply(i, corners[i], warped_imgs_uchar[i], this->masks[i]);
    TIMER("Compensator apply");

    SAVE_MAT_VEC("warped_img_compensator", warped_imgs_uchar);
    SAVE_MAT_VEC("warped_mask_compensator", masks_scale);

    std::vector<cv::UMat> masks_seam(inputs.size());
    // TODO GraphCut and DpSeamFinder has bugs?
    // cv::Ptr<cv::detail::SeamFinder> seam_finder = new cv::detail::GraphCutSeamFinder(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
    // cv::Ptr<cv::detail::SeamFinder> seam_finder = new cv::detail::DpSeamFinder(cv::detail::DpSeamFinder::COLOR);
    cv::Ptr<cv::detail::SeamFinder> seam_finder = new cv::detail::VoronoiSeamFinder();
    seam_finder->find(warped_imgs_uchar_scale, corners, masks_scale);
    for(int i = 0 ; i < inputs.size() ; i += 1)
        cv::resize(masks_scale[i], masks_seam[i], cv::Size(), 1.0/scale, 1.0/scale);
    TIMER("Seam finder");
    // TODO dilate mask?

    SAVE_MAT_VEC("warped_mask_seam", masks_seam);

    cv::Ptr<cv::detail::Blender> blender = 
        cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, true);
    // auto blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, false);
    // TODO set band number
    dynamic_cast<cv::detail::MultiBandBlender *>(static_cast<cv::detail::Blender *>(blender))->setNumBands(9);
    blender->prepare(corners, sizes);
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        cv::UMat m;
        warped_imgs_uchar[i].convertTo(m, CV_16SC3);
        blender->feed(m, masks_seam[i], corners[i]);
    }

    cv::UMat result, result_mask;
    blender->blend(result, result_mask);
    TIMER("Blender");

    result.convertTo(output, CV_8UC3);
}
