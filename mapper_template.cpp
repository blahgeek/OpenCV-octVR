/* 
* @Author: BlahGeek
* @Date:   2015-12-07
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-07
*/

#include <iostream>
#include "./libmap.hpp"
#include "./camera.hpp"
#include <stdio.h>
#include <opencv2/stitching/detail/seam_finders.hpp>

using namespace vr;

MapperTemplate::MapperTemplate(const std::string & to,
                               const json & to_opts,
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
                               const json & from_opts) {
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

    this->map1s.push_back(map1);
    this->map2s.push_back(map2);
    this->masks.push_back(mask);
}

static const char * DUMP_MAGIC = "VRv02";

void MapperTemplate::dump(std::ofstream & f) {
    if(this->seam_masks.empty()) {
        // VoronoiSeamFinder does not care about images
        cv::Ptr<cv::detail::SeamFinder> seam_finder = new cv::detail::VoronoiSeamFinder();

        std::vector<cv::UMat> srcs(masks.size()), umasks(masks.size());
        for(int i = 0 ; i < masks.size() ; i += 1) {
            srcs[i].create(masks[i].size(), CV_8UC3);
            masks[i].copyTo(umasks[i]);
        }
        seam_finder->find(srcs, 
                          std::vector<cv::Point2i>(masks.size(), cv::Point2i(0, 0)),
                          umasks);

        this->seam_masks.resize(masks.size());
        for(int i = 0 ; i < masks.size() ; i += 1)
            umasks[i].copyTo(seam_masks[i]);
    }

    f.write(DUMP_MAGIC, strlen(DUMP_MAGIC));

    auto W64i = [&](int64_t x) {
        f.write(reinterpret_cast<char *>(&x), sizeof(int64_t));
    };

    W64i(out_size.width);
    W64i(out_size.height);

    W64i(map1s.size());
    for(int i = 0 ; i < map1s.size() ; i += 1) {
        assert(map1s[i].type() == CV_32FC1 && map1s[i].size() == out_size);
        assert(map2s[i].type() == CV_32FC1 && map2s[i].size() == out_size);
        assert(masks[i].type() == CV_8UC1 && masks[i].size() == out_size);
        assert(seam_masks[i].type() == CV_8UC1 && seam_masks[i].size() == out_size);

        for(int k = 0 ; k < out_size.height ; k += 1) {
            f.write(map1s[i].ptr<char>(k), out_size.width * 4);
            f.write(map2s[i].ptr<char>(k), out_size.width * 4);
            f.write(masks[i].ptr<char>(k), out_size.width);
            f.write(seam_masks[i].ptr<char>(k), out_size.width);
        }
    }
}

MapperTemplate::MapperTemplate(std::ifstream & f) {
    char read_magic[16];
    f.read(read_magic, strlen(DUMP_MAGIC));
    if(strncmp(read_magic, DUMP_MAGIC, strlen(DUMP_MAGIC)) != 0)
        throw std::string("Invalid data file");

    auto R64i = [&]() -> int64_t {
        int64_t _ret = 0;
        f.read(reinterpret_cast<char *>(&_ret), sizeof(int64_t));
        return _ret;
    };

    this->out_size.width = R64i();
    this->out_size.height = R64i();

    int in_count = R64i();
    this->map1s.resize(in_count);
    this->map2s.resize(in_count);
    this->masks.resize(in_count);
    this->seam_masks.resize(in_count);
    for(int i = 0 ; i < in_count ; i += 1) {
        for(int k = 0 ; k < out_size.height ; k += 1) {
            map1s[i].create(out_size, CV_32FC1);
            map2s[i].create(out_size, CV_32FC1);
            masks[i].create(out_size, CV_8UC1);
            seam_masks[i].create(out_size, CV_8UC1);
            f.read(map1s[i].ptr<char>(k), out_size.width * 4);
            f.read(map2s[i].ptr<char>(k), out_size.width * 4);
            f.read(masks[i].ptr<char>(k), out_size.width);
            f.read(seam_masks[i].ptr<char>(k), out_size.width);
        }
    }
}
