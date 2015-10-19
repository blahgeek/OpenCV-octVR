/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-19
*/

#include <iostream>
#include "./libmap.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "./equirectangular.hpp"
#include "./normal.hpp"
#include "./pinhole_cam.hpp"
#include "./fisheye_cam.hpp"

using namespace vr;

std::unique_ptr<Map> Map::New(const std::string & type, const json & options) {
    #define X(s, t) \
        else if (type == s) return std::unique_ptr<Map>(new t(options));

    if(false){}

    X("normal", Normal)
    X("pinhole", PinholeCamera)
    X("fisheye", FisheyeCamera)
    X("equirectangular", Equirectangular)

    return nullptr;

    #undef X
}

Map::Map(const json & options) {
    this->rotate_vector = std::vector<double>({0, 0, 0});
    if(options.find("rotate") != options.end())
        this->rotate_vector = options["rotate"].get<std::vector<double>>();
    cv::Rodrigues(rotate_vector, this->rotate_matrix);
}

cv::Point2d Map::sphere_xyz_to_lonlat(const cv::Point3d & xyz) {
    auto p = xyz * (1.0 / cv::norm(xyz));
    return cv::Point2d(atan2(p.z, p.x), asin(p.y));
}

cv::Point3d Map::sphere_lonlat_to_xyz(const cv::Point2d & lonlat) {
    auto lon = lonlat.x;
    auto lat = lonlat.y;
    return cv::Point3d(cos(lon) * cos(lat),
                       sin(lat),
                       sin(lon) * cos(lat));
}

std::vector<cv::Point2d> Map::obj_to_image(const std::vector<cv::Point2d> & lonlats) {
    // convert lon/lat to xyz in sphere
    cv::Mat obj_points(3, lonlats.size(), CV_64FC1);
    for(int i = 0 ; i < lonlats.size() ; i += 1) {
        auto xyz = sphere_lonlat_to_xyz(lonlats[i]);
        obj_points.at<double>(0, i) = xyz.x;
        obj_points.at<double>(1, i) = xyz.y;
        obj_points.at<double>(2, i) = xyz.z;
    }

    // rotate it
    obj_points = rotate_matrix * obj_points;

    std::vector<cv::Point2d> ret;
    ret.reserve(lonlats.size());

    for(int i = 0 ; i < lonlats.size() ; i += 1) {
        cv::Mat xyz = obj_points.col(i);
        auto lonlat = sphere_xyz_to_lonlat(cv::Point3d(xyz));
        ret.push_back(this->obj_to_image_single(lonlat));
    }
    return ret;
}

std::vector<cv::Point2d> Map::image_to_obj(const std::vector<cv::Point2d> & xys) {
    cv::Mat obj_points(3, xys.size(), CV_64FC1);

    for(int i = 0 ; i < xys.size() ; i += 1) {
        auto lonlat = this->image_to_obj_single(xys[i]);
        auto xyz = sphere_lonlat_to_xyz(lonlat);
        obj_points.at<double>(0, i) = xyz.x;
        obj_points.at<double>(1, i) = xyz.y;
        obj_points.at<double>(2, i) = xyz.z;
    }

    // rotate it
    obj_points = rotate_matrix.inv() * obj_points;

    // convert xyz back to lonlat
    std::vector<cv::Point2d> ret;
    ret.reserve(xys.size());
    for(int i = 0 ; i < xys.size() ; i += 1) {
        cv::Mat xyz = obj_points.col(i);
        ret.push_back(sphere_xyz_to_lonlat(cv::Point3d(xyz)));
    }

    return ret;
}

Remapper::Remapper(const std::string & from, const json & from_opts,
                   const std::string & to, const json & to_opts,
                   int in_width, int in_height, int out_width, int out_height):
in_width(in_width), in_height(in_height), out_width(out_width), out_height(out_height) {
    this->in_map = Map::New(from, from_opts);
    this->out_map = Map::New(to, to_opts);
    if(!this->in_map || !this->out_map)
        throw std::string("Invalid map type");

    if(this->out_height <= 0 && this->out_width <= 0)
        throw std::string("Output width/height invalid");
    double output_aspect_ratio = this->out_map->get_aspect_ratio();
    if(this->out_height <= 0)
        this->out_height = int(double(this->out_width) / output_aspect_ratio);
    if(this->out_width <= 0)
        this->out_width = int(double(this->out_height) * output_aspect_ratio);
    std::cerr << "Output size: " << this->out_width << "x" << this->out_height << std::endl;

    std::vector<cv::Point2d> points;
    for(int j = 0 ; j < this->out_height ; j += 1)
        for(int i = 0 ; i < this->out_width ; i += 1)
            points.push_back(cv::Point2d(double(i) / this->out_width,
                                         double(j) / this->out_height));

    std::vector<cv::Point2d> tmp = this->out_map->image_to_obj(points);
    tmp = this->in_map->obj_to_image(tmp);

    this->map_cache.reserve(points.size());
    for(auto & p: tmp) {
        double x = p.x * this->in_width;
        double y = p.y * this->in_height;
        if(x < 0 || x >= this->in_width)
            x = NAN;
        if(y < 0 || y >= this->in_height)
            x = NAN;
        this->map_cache.push_back(cv::Point2d(x, y));
    }
}

MultiMapper::MultiMapper(const std::string & to, const json & to_opts,
                         int out_width, int out_height) {
    this->out_map = Map::New(to, to_opts);
    if(!this->out_map)
        throw std::string("Invalid output map type");

    if(out_height <= 0 && out_width <= 0)
        throw std::string("Output width/height invalid");
    double output_aspect_ratio = this->out_map->get_aspect_ratio();
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
    this->output_map_points = this->out_map->image_to_obj(tmp);
}

void MultiMapper::add_input(const std::string & from, const json & from_opts,
                            int in_width, int in_height) {
    this->in_sizes.push_back(cv::Size(in_width, in_height));
    auto _map = Map::New(from, from_opts);
    if(!_map)
        throw std::string("Invalid input map type");
    this->in_maps.push_back(std::move(_map));

    auto tmp = this->in_maps.back()->obj_to_image(this->output_map_points);

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

std::pair<int, cv::Point2d> MultiMapper::get_map(int w, int h) {
    for(int i = 0 ; i < this->in_maps.size() ; i += 1) {
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
