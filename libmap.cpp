/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-23
*/

#include <iostream>
#include "./libmap_impl.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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

    std::vector<cv::Point2d> map_cache;
    map_cache.reserve(tmp.size());
    for(auto & p: tmp) {
        double x = p.x * in_width;
        double y = p.y * in_height;
        if(x < 0 || x >= in_width)
            x = NAN;
        if(y < 0 || y >= in_height)
            y = NAN;
        map_cache.push_back(cv::Point2d(x, y));
    }
    this->map_caches.push_back(map_cache);
}

std::pair<int, cv::Point2d> MultiMapperImpl::get_map(int w, int h) {
    for(int i = 0 ; i < this->in_cameras.size() ; i += 1) {
        if(w < 0 || w >= this->out_size.width ||
           h < 0 || h >= this->out_size.height)
            continue;
        int index = h * this->out_size.width + w;
        auto p = this->map_caches[i][index];
        if(isnan(p.x) || isnan(p.y))
            continue;

        return std::make_pair(i, p);
    }
    return std::make_pair(0, cv::Point2d(NAN, NAN));
}

void MultiMapperImpl::get_output(const std::vector<cv::Mat> & inputs, cv::Mat & output) {
    for(int i = 0 ; i < inputs.size() ; i += 1) {
        assert(inputs[i].type() == CV_8UC3);
        assert(inputs[i].size() == in_sizes[i]);
    }
    assert(output.type() == CV_8UC3 && output.size() == this->out_size);

    for(int j = 0 ; j < out_size.height ; j += 1) {
        for(int i = 0 ; i < out_size.width ; i += 1) {
            auto map = get_map(i, j);
            if(isnan(map.second.x) || isnan(map.second.y))
                continue;
            const auto & input = inputs[map.first];
            for(int k = 0 ; k < 3 ; k += 1)
                output.at<unsigned char>(j, i*3 + k) = 
                    input.at<unsigned char>(floor(map.second.y),
                                            floor(map.second.x) * 3 + k);
        }
    }
}
