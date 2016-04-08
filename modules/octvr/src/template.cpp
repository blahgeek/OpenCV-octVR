/* 
* @Author: BlahGeek
* @Date:   2015-12-07
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-08
*/

#include "octvr.hpp"
#include "./camera.hpp"
#include <iostream>
#include <stdio.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "parallel_caller.hpp"


using namespace vr;

MapperTemplate::MapperTemplate(const std::string & to,
                               const rapidjson::Value & to_opts,
                               int width, int height):
out_type(to), out_opts(&to_opts) {

    std::unique_ptr<Camera> out_camera = Camera::New(out_type, *out_opts);
    if(!out_camera)
        throw std::string("Invalid output camera type");

    if((height <= 0 && width <= 0) || (height > 0 && width > 0))
        throw std::string("Output width/height invalid");
    double output_aspect_ratio = out_camera->get_aspect_ratio();
    if(height <= 0)
        height = int(double(width) / output_aspect_ratio);
    if(width <= 0)
        width = int(double(height) * output_aspect_ratio);
    std::cerr << "Output type: " << to << ", Size: " << width << "x" << height << std::endl;
    this->out_size = cv::Size(width, height);
    this->visible_mask = std::vector<bool>(width * height, false);
}

void MapperTemplate::add_input(const std::string & from,
                               const rapidjson::Value & from_opts,
                               bool overlay,
                               bool use_roi) {
    std::unique_ptr<Camera> out_camera = Camera::New(out_type, *out_opts);
    std::unique_ptr<Camera> cam = Camera::New(from, from_opts);
    if(!cam)
        throw std::string("Invalid input camera type");

    std::vector<cv::Point2d> out_points;
    for(int j = 0 ; j < out_size.height ; j += 1)
        for(int i = 0 ; i < out_size.width ; i += 1)
            out_points.push_back(cv::Point2d(double(i) / out_size.width,
                                             double(j) / out_size.height));
    auto output_map_points = out_camera->image_to_obj(out_points);

    auto tmp = cam->obj_to_image(output_map_points);
    auto visible_tmp = cam->get_include_mask(output_map_points);

    int min_h = out_size.height, max_h = 0;
    int min_w = out_size.width, max_w = 0;

    cv::Mat map1(out_size, CV_32FC1), map2(out_size, CV_32FC1);
    cv::Mat mask(out_size, CV_8U);
    auto process_row_block = [&](const cv::Range& row_range)
    {
        for(int h = row_range.start; h < row_range.end; h += 1) {
            unsigned char * mask_row = mask.ptr(h);
            float * map1_row = map1.ptr<float>(h);
            float * map2_row = map2.ptr<float>(h);

            // auto process_point_block = [&](const cv::Range& r)
            // {
                for (int w = 0; w < out_size.width; w += 1)
                {
                    auto index = w + out_size.width * h;
                    float x = tmp[index].x;
                    float y = tmp[index].y;
                    if(isnan(x) || isnan(y) ||
                       x < 0 || x >= 1.0f || y < 0 || y >= 1.0f ||  // out of border, should be black when doing remap()
                       visible_mask[index])                         // or this point has been selected to be visible by other image (not override)
                    {
                        mask_row[w] = 0;
                        map1_row[w] = map2_row[w] = -1.0;
                    }
                    else {
                        mask_row[w] = 255;
                        map1_row[w] = x;
                        map2_row[w] = y;

                        // update ROI
                        if(h < min_h) min_h = h;
                        if(h > max_h) max_h = h;
                        if(w < min_w) min_w = w;
                        if(w > max_w) max_w = w;
                    }
                    // Update visible_mask and other inputs' masks.
                    if (visible_tmp.empty())
                        continue;
                    if (!visible_mask[index] && visible_tmp[index])
                        for (auto & prior_input : this->inputs) {
                            // When using ROI, the prior inputs' masks are not full-size
                            // which causes overflow
                            auto prior_roi = prior_input.roi;
                            if (h < prior_roi.y || h >= prior_roi.y + prior_roi.height ||
                                w < prior_roi.x || w >= prior_roi.x + prior_roi.width )
                                continue;
                            unsigned char * prior_mask_row = prior_input.mask.ptr(h - prior_roi.y);
                            prior_mask_row[w - prior_roi.x] = 0;
                        }
                    visible_mask[index] = visible_mask[index] || visible_tmp[index];
                }
            // };
            // parallel_for_caller(cv::Range(0, out_size.width), process_point_block);
        }
    };
    parallel_for_caller(cv::Range(0, out_size.height), process_row_block);

    CV_Assert(min_h <= max_h && min_w <= max_w);

    if(min_w > 0) min_w -= 1;
    if(min_h > 0) min_h -= 1;
    if(max_w < out_size.width - 1) max_w += 1;
    if(max_h < out_size.height - 1) max_h += 1;

    cv::Rect roi(min_w, min_h, max_w + 1 - min_w, max_h + 1 - min_h);
    if(!use_roi)
        roi = cv::Rect(0, 0, out_size.width, out_size.height);

    MapperTemplate::Input input;
    input.map1 = map1(roi);
    input.map2 = map2(roi);
    input.mask = mask(roi);
    input.roi = roi;

    std::cerr << "ROI: " << roi << std::endl;

    if(overlay)
        this->overlay_inputs.push_back(input);
    else
        this->inputs.push_back(input);
}

void MapperTemplate::create_masks(const std::vector<cv::Mat> & imgs) {
    std::vector<cv::UMat> srcs(inputs.size()), umasks(inputs.size());

    double scale = std::min(1.0, 960.0 / out_size.width);
    std::cerr << "Scale for creating mask: " << scale << std::endl;

    std::vector<cv::Rect> scaled_rois;
    std::vector<cv::Point> scaled_corners;

    for(size_t i = 0 ; i < inputs.size() ; i += 1) {
        cv::Rect scaled_roi(inputs[i].roi.x * scale,
                            inputs[i].roi.y * scale,
                            inputs[i].roi.width * scale,
                            inputs[i].roi.height * scale);
        scaled_rois.push_back(scaled_roi);
        scaled_corners.push_back(scaled_roi.tl());

        if(i < imgs.size()) {
            cv::Mat tmp0, tmp1;
            cv::remap(imgs[i], tmp0,
                      inputs[i].map1 * imgs[i].cols,
                      inputs[i].map2 * imgs[i].rows, cv::INTER_LINEAR);
            tmp0.convertTo(tmp1, CV_32FC3, 1.0/255.0);
            cv::resize(tmp1, srcs[i], scaled_roi.size());
        } else {
            srcs[i].create(scaled_roi.size(), CV_8UC3);
        }
        cv::resize(inputs[i].mask, umasks[i], scaled_roi.size());
    }

    cv::detail::SeamFinder * seam_finder = nullptr;
    if(imgs.empty()) {
        // VoronoiSeamFinder do not care about image content
        // std::cerr << "Using voronoi seam finder..." << std::endl;
        // seam_finder = new cv::detail::VoronoiSeamFinder();
        std::cerr << "Using BFS seam finder..." << std::endl;
        seam_finder = new cv::detail::BFSSeamFinder();
    }
    else {
        std::cerr << "Using graph cut seam finder..." << std::endl;
        seam_finder = new cv::detail::GraphCutSeamFinder();
    }
    seam_finder->find(srcs, scaled_corners, umasks);

    this->seam_masks.resize(inputs.size());
    for(size_t i = 0 ; i < inputs.size() ; i += 1)
        cv::resize(umasks[i], seam_masks[i], inputs[i].roi.size());

    delete seam_finder;
}

static const char * DUMP_MAGIC = "VRv10";

void MapperTemplate::dump(std::ofstream & f) {
    if(this->seam_masks.empty())
        this->create_masks();

    f.write(DUMP_MAGIC, strlen(DUMP_MAGIC));

    auto W64i = [&](int64_t x) {
        f.write(reinterpret_cast<char *>(&x), sizeof(int64_t));
    };

    auto WRect = [&](cv::Rect r) {
        W64i(r.x); W64i(r.y); W64i(r.width); W64i(r.height);
    };

    auto Wmat = [&](cv::Mat & m) {
        W64i(m.type());
        W64i(m.rows);
        W64i(m.cols);
        int elem_size = m.elemSize();
        for(int k = 0 ; k < m.rows ; k += 1)
            f.write(m.ptr<char>(k), m.cols * elem_size);
    };

    W64i(out_size.width);
    W64i(out_size.height);

    W64i(inputs.size());
    for(auto & input: inputs) {
        WRect(input.roi);
        Wmat(input.map1);
        Wmat(input.map2);
        Wmat(input.mask);
    }
    assert(inputs.size() == seam_masks.size());
    for(auto & m: seam_masks)
        Wmat(m);

    W64i(overlay_inputs.size());
    for(auto & input: overlay_inputs) {
        WRect(input.roi);
        Wmat(input.map1);
        Wmat(input.map2);
        Wmat(input.mask);
    }
}

MapperTemplate::MapperTemplate(std::ifstream & f) {
    char read_magic[16];
    f.read(read_magic, strlen(DUMP_MAGIC));
    if(strncmp(read_magic, DUMP_MAGIC, strlen(DUMP_MAGIC)) != 0)
        throw std::string("Invalid data file (version does not match)");

    auto R64i = [&]() -> int64_t {
        int64_t _ret = 0;
        f.read(reinterpret_cast<char *>(&_ret), sizeof(int64_t));
        return _ret;
    };

    auto RRect = [&]() -> cv::Rect {
        auto x = R64i();
        auto y = R64i();
        auto width = R64i();
        auto height = R64i();
        return cv::Rect(x, y, width, height);
    };

    auto Rmat = [&]() -> cv::Mat {
        int type = R64i();
        int rows = R64i();
        int cols = R64i();
        cv::Mat ret(rows, cols, type);
        int elem_size = ret.elemSize();
        for(int k = 0 ; k < rows ; k += 1)
            f.read(ret.ptr<char>(k), cols * elem_size);
        return ret;
    };

    this->out_size.width = R64i();
    this->out_size.height = R64i();

    this->inputs.resize(R64i());
    for(auto & input: inputs) {
        input.roi = RRect();
        input.map1 = Rmat();
        input.map2 = Rmat();
        input.mask = Rmat();
    }
    this->seam_masks.resize(this->inputs.size());
    for(size_t i = 0 ; i < this->seam_masks.size() ; i += 1)
        this->seam_masks[i] = Rmat();

    this->overlay_inputs.resize(R64i());
    for(auto & input: overlay_inputs) {
        input.roi = RRect();
        input.map1 = Rmat();
        input.map2 = Rmat();
        input.mask = Rmat();
    }
}
