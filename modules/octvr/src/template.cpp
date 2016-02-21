/* 
* @Author: BlahGeek
* @Date:   2015-12-07
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-21
*/

#include "octvr.hpp"
#include "./camera.hpp"
#include <iostream>
#include <stdio.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"


using namespace vr;

MapperTemplate::MapperTemplate(const std::string & to,
                               const rapidjson::Value & to_opts,
                               int width, int height):
out_type(to), out_opts(to_opts) {

    std::unique_ptr<Camera> out_camera = Camera::New(out_type, out_opts);
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
}

void MapperTemplate::add_input(const std::string & from,
                               const rapidjson::Value & from_opts,
                               bool overlay) {
    std::unique_ptr<Camera> out_camera = Camera::New(out_type, out_opts);
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
    cv::Mat map1(out_size, CV_32FC1), map2(out_size, CV_32FC1);
    cv::Mat mask(out_size, CV_8U);
    for(int h = 0 ; h < out_size.height ; h += 1) {
        unsigned char * mask_row = mask.ptr(h);
        float * map1_row = map1.ptr<float>(h);
        float * map2_row = map2.ptr<float>(h);

        for(int w = 0 ; w < out_size.width ; w += 1) {
            auto index = w + out_size.width * h;
            float x = tmp[index].x;
            float y = tmp[index].y;
            if(isnan(x) || isnan(y) ||
               x < 0 || x >= 1.0f || y < 0 || y >= 1.0f) {
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

    MapperTemplate::Input input;
    input.map1 = map1;
    input.map2 = map2;
    input.mask = mask;

    if(overlay)
        this->overlay_inputs.push_back(input);
    else
        this->inputs.push_back(input);
}

void MapperTemplate::create_masks(const std::vector<cv::Mat> & imgs) {
    std::vector<cv::UMat> srcs(inputs.size()), umasks(inputs.size());

    cv::Size scaled_size = this->out_size;
    double scale = std::min(1.0, 800.0 / out_size.width);
    scaled_size.width *= scale;
    scaled_size.height *= scale;
    std::cerr << "Scaled size: " << scaled_size << std::endl;

    for(size_t i = 0 ; i < inputs.size() ; i += 1) {
        if(i < imgs.size()) {
            cv::Mat tmp0, tmp1;
            cv::remap(imgs[i], tmp0, inputs[i].map1, inputs[i].map2, cv::INTER_LINEAR);
            tmp0.convertTo(tmp1, CV_32FC3, 1.0/255.0);
            cv::resize(tmp1, srcs[i], scaled_size);
        }
        else
            srcs[i].create(scaled_size, CV_8UC3);
        cv::resize(inputs[i].mask, umasks[i], scaled_size);
    }

    cv::detail::SeamFinder * seam_finder = nullptr;
    if(imgs.empty()) {
        // VoronoiSeamFinder do not care about image content
        //std::cerr << "Using voronoi seam finder..." << std::endl;
        //seam_finder = new cv::detail::VoronoiSeamFinder();
        std::cerr << "Using BFS seam finder..." << std::endl;
        seam_finder = new cv::detail::BFSSeamFinder();
    }
    else {
        std::cerr << "Using graph cut seam finder..." << std::endl;
        seam_finder = new cv::detail::GraphCutSeamFinder();
    }
    seam_finder->find(srcs,
                      std::vector<cv::Point2i>(inputs.size(), cv::Point2i(0, 0)),
                      umasks);

    this->seam_masks.resize(inputs.size());
    for(size_t i = 0 ; i < inputs.size() ; i += 1)
        cv::resize(umasks[i], seam_masks[i], this->out_size);

    delete seam_finder;
}

static const char * DUMP_MAGIC = "VRv03";

void MapperTemplate::dump(std::ofstream & f) {
    if(this->seam_masks.empty())
        this->create_masks();

    f.write(DUMP_MAGIC, strlen(DUMP_MAGIC));

    auto W64i = [&](int64_t x) {
        f.write(reinterpret_cast<char *>(&x), sizeof(int64_t));
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
        Wmat(input.map1);
        Wmat(input.map2);
        Wmat(input.mask);
    }
    assert(inputs.size() == seam_masks.size());
    for(auto & m: seam_masks)
        Wmat(m);

    W64i(overlay_inputs.size());
    for(auto & input: overlay_inputs) {
        Wmat(input.map1);
        Wmat(input.map2);
        Wmat(input.mask);
    }
}

static const rapidjson::Value __unused__;

MapperTemplate::MapperTemplate(std::ifstream & f): out_opts(__unused__) {
    char read_magic[16];
    f.read(read_magic, strlen(DUMP_MAGIC));
    if(strncmp(read_magic, DUMP_MAGIC, strlen(DUMP_MAGIC)) != 0)
        throw std::string("Invalid data file");

    auto R64i = [&]() -> int64_t {
        int64_t _ret = 0;
        f.read(reinterpret_cast<char *>(&_ret), sizeof(int64_t));
        return _ret;
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
        input.map1 = Rmat();
        input.map2 = Rmat();
        input.mask = Rmat();
    }
    this->seam_masks.resize(this->inputs.size());
    for(size_t i = 0 ; i < this->seam_masks.size() ; i += 1)
        this->seam_masks[i] = Rmat();

    this->overlay_inputs.resize(R64i());
    for(auto & input: overlay_inputs) {
        input.map1 = Rmat();
        input.map2 = Rmat();
        input.mask = Rmat();
    }
}
