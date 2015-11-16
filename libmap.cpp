/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-16
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

#include <sys/time.h>

// TODO Set by options
#define WORKING_MEGAPIX 0.1
#define BLENDER_STRENGTH 0.05

using namespace vr;

static int64_t gettime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

Timer::Timer(std::string name): t(gettime()), name(name) {}
Timer::Timer(): Timer::Timer("") {}

void Timer::tick(std::string msg) {
    auto tn = gettime();
    std::cerr << "[ Timer " << name << "] " << msg << ": " << (tn - t) / 1000.0 << "ms" << std::endl;
    t = tn;
}

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

    cv::Size in_size(in_width, in_height);
    this->in_sizes.push_back(in_size);
    std::unique_ptr<Camera> cam = Camera::New(from, from_opts);
    if(!cam)
        throw std::string("Invalid input camera type");

    auto tmp = cam->obj_to_image(this->output_map_points);
    cv::Mat map1(out_size, CV_32FC1), map2(out_size, CV_32FC1);
    cv::Mat mask(out_size, CV_8U);
    for(int h = 0 ; h < out_size.height ; h += 1) {
        unsigned char * mask_row = mask.ptr(h);
        float * map1_row = map1.ptr<float>(h);
        float * map2_row = map2.ptr<float>(h);

        for(int w = 0 ; w < out_size.width ; w += 1) {
            auto index = w + out_size.width * h;
            float x = tmp[index].x * in_width;
            float y = tmp[index].y * in_height;
            if(isnan(x) || isnan(y) ||
               x < 0 || x >= in_width || y < 0 || y >= in_height) {
                mask_row[w] = 0;
                map1_row[w] = map2_row[w] = -1.0; // out of border, should be black when doing remap()
            }
            else {
                mask_row[w] = 255;
                map1_row[w] = x;
                map2_row[w] = y;
            }
        }
    }

    GpuMat map1_u, map2_u;
    map1_u.upload(map1); map2_u.upload(map2);
    this->map1s.push_back(map1_u);
    this->map2s.push_back(map2_u);
    GpuMat mask_u;
    mask_u.upload(mask);
    this->masks.push_back(mask_u);

    double working_scale = std::min(1.0, sqrt(WORKING_MEGAPIX * 1e6 / in_size.area()));
    this->working_scales.push_back(working_scale);

    GpuMat scaled_mask;
    cv::cuda::resize(mask_u, scaled_mask, cv::Size(), working_scale, working_scale);
    this->scaled_masks.push_back(scaled_mask);
}

void MultiMapperImpl::prepare() {
    Timer timer("MultiMapper Prepare");

    std::vector<cv::UMat> host_masks(masks.size());
    for(int i = 0 ; i < masks.size() ; i += 1)
        masks[i].download(host_masks[i]);
    timer.tick("Download masks");

    // VoronoiSeamFinder does not care about images
    cv::Ptr<cv::detail::SeamFinder> seam_finder = new cv::detail::VoronoiSeamFinder();
    std::vector<cv::UMat> srcs;
    for(auto & m: host_masks)
        srcs.push_back(cv::UMat(m.size(), CV_8UC3));
    seam_finder->find(srcs,
                      std::vector<cv::Point2i>(masks.size(), cv::Point2i(0, 0)), 
                      host_masks);
    timer.tick("Seam finder");
    // TODO dilate mask?

    seam_masks.resize(masks.size());
    for(int i = 0 ; i < masks.size() ; i += 1)
        seam_masks[i].upload(host_masks[i]);
    timer.tick("Upload seam masks");
}

#ifdef DEBUG_SAVE_MAT

#define SAVE_MAT(N, S, MAT) \
    do { \
        char tmp[64]; \
        snprintf(tmp, 64, S "_%d.jpg", N);\
        cv::Mat __local_mat;
        MAT.download(__local_mat);
        imwrite(tmp, __local_mat); \
    } while(false)

#define SAVE_MAT_VEC(S, MATS) \
    do { \
        for(int __i = 0 ; __i < MATS.size() ; __i += 1) \
            SAVE_MAT(__i, S, MATS[__i]); \
    } while(false)

#else

#define SAVE_MAT_VEC(S, MATS)

#endif

void MultiMapperImpl::get_output(const std::vector<cv::Mat> & inputs, cv::Mat & output) {
    Timer timer("MultiMapper");

    for(int i = 0 ; i < inputs.size() ; i += 1) {
        assert(inputs[i].type() == CV_8UC3);
        assert(inputs[i].size() == in_sizes[i]);
    }
    assert(output.type() == CV_8UC3 && output.size() == this->out_size);

    std::vector<GpuMat> gpu_inputs(inputs.size());
    for(int i = 0 ; i < inputs.size() ; i += 1)
        gpu_inputs[i].upload(inputs[i]);
    timer.tick("Upload inputs to GPU");
    
    std::vector<cv::Point2i> corners;
    std::vector<cv::Size> sizes;
    std::vector<GpuMat> warped_imgs_uchar(inputs.size());
    
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        cv::cuda::remap(gpu_inputs[i], warped_imgs_uchar[i], map1s[i], map2s[i], CV_INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar_<unsigned char>(0, 0, 0) /* stream */);
        corners.emplace_back(0, 0);
        sizes.push_back(out_size);
    }
    timer.tick("Remapping images");

    SAVE_MAT_VEC("warped_img", warped_imgs_uchar);
    SAVE_MAT_VEC("warped_mask", masks);

    std::vector<GpuMat> warped_imgs_uchar_scale(inputs.size());
    std::vector<GpuMat> scaled_masks_clone(inputs.size());
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        cv::cuda::resize(warped_imgs_uchar[i], warped_imgs_uchar_scale[i], cv::Size(), 
                         working_scales[i], working_scales[i]);
        // cv::erode(scaled_masks[i], scaled_masks_clone[i], cv::Mat(), cv::Point(-1, -1), 3);
        scaled_masks[i].copyTo(scaled_masks_clone[i]);
    }
    timer.tick("Scale");

    // if(this->compensator.empty()) {
    //     std::cerr << "Re-computing compensator gain" << std::endl;
    //     // GAIN_BLOCKS split every image into multiple blocks to do exposure compensator
    //     // That is slow and not good(?) for VR video
    //     this->compensator = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN);
    //     compensator->feed(corners, warped_imgs_uchar_scale, scaled_masks_clone);
    //     timer.tick("Compensator");
    // }

    // TODO Optimize this
    // GainCompensator::apply does img *= gain for every image
    // while size of image is large (output size), and only part of it is valid (mask)
    // for(int i = 0 ; i < inputs.size() ; i += 1)
    //     compensator->apply(i, corners[i], warped_imgs_uchar[i], this->masks[i]);
    // timer.tick("Compensator apply");

    SAVE_MAT_VEC("warped_img_compensator", warped_imgs_uchar);
    SAVE_MAT_VEC("warped_mask_compensator", scaled_masks);

    double blend_width = sqrt(out_size.area() * 1.0f) * BLENDER_STRENGTH;
    int blend_bands = int(ceil(log(blend_width)/log(2.)) - 1.);
    std::cerr << "Using MultiBandBlender with band number = " << blend_bands << std::endl;
    auto blender = cv::makePtr<cv::detail::MultiBandGPUBlender>(out_size, blend_bands);
    for(int i = 0 ; i < inputs.size() ; i += 1)
        blender->feed(warped_imgs_uchar[i], seam_masks[i]);

    timer.tick("Blender prepare");

    GpuMat result;
    blender->blend(result);
    timer.tick("Blender blend");

    assert(result.type() == CV_8UC3);
    result.download(output);
    timer.tick("Convert result");
}

void MultiMapperImpl::get_single_output(const cv::Mat & input, cv::Mat & output) {
    assert(input.type() == CV_8UC3);
    assert(output.type() == CV_8UC3 && output.size() == this->out_size);
    assert(this->in_sizes.size() > 0);

    Timer timer("MultiMapper");

    cv::remap(input, output, map1s[0], map2s[0], CV_INTER_LINEAR);

    timer.tick("Remapping single image");
}
