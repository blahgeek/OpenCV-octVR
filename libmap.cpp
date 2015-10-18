/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-18
*/

#include <iostream>
#include "./libmap.hpp"

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

std::vector<std::tuple<double, double, bool>>
    Map::lonlat_to_xy_batch(const std::vector<std::pair<double, double>> & points) {

    std::vector<std::tuple<double, double, bool>> ret;
    for(auto & point: points) {
        bool valid = true;
        std::pair<double, double> xy;
        try {
            xy = this->lonlat_to_xy(point.first, point.second);
        } catch (OutOfRange & e) {
            valid = false;
        }
        ret.push_back(std::make_tuple(xy.first, xy.second, valid));
    }
    return ret;
}

std::vector<std::tuple<double, double, bool>>
    Map::xy_to_lonlat_batch(const std::vector<std::pair<double, double>> & points) {

    std::vector<std::tuple<double, double, bool>> ret;
    for(auto & point: points) {
        bool valid = true;
        std::pair<double, double> lonlat;
        try {
            lonlat = this->xy_to_lonlat(point.first, point.second);
        } catch (OutOfRange & e) {
            valid = false;
        }
        ret.push_back(std::make_tuple(lonlat.first, lonlat.second, valid));
    }
    return ret;
}

Remapper::Remapper(const std::string & from, const json & from_opts,
                   const std::string & to, const json & to_opts,
                   double rotate_z, double rotate_y, double rotate_x,
                   int in_width, int in_height, int out_width, int out_height):
rotate_z(rotate_z), rotate_y(rotate_y), rotate_x(rotate_x),
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

    std::vector<std::pair<double, double>> points;
    for(int j = 0 ; j < this->out_height ; j += 1)
        for(int i = 0 ; i < this->out_width ; i += 1)
            points.push_back(std::make_tuple(double(i) / this->out_width,
                                             double(j) / this->out_height));
    this->map_cache = this->out_map->xy_to_lonlat_batch(points);

    std::vector<bool> internal_valids;
    points.clear();
    for(auto & x: this->map_cache) {
        auto newlonlat = this->rotate(std::get<0>(x), std::get<1>(x));
        points.push_back(newlonlat);
        internal_valids.push_back(std::get<2>(x));
    }
    this->map_cache = this->in_map->lonlat_to_xy_batch(points);

    for(int i = 0 ; i < this->map_cache.size() ; i += 1) {
        double x = std::get<0>(this->map_cache[i]) * this->in_width;
        double y = std::get<1>(this->map_cache[i]) * this->in_height;
        bool valid = std::get<2>(this->map_cache[i]);
        valid &= internal_valids[i];

        if(isnan(x) || x < 0 || x >= this->in_width ||
           isnan(y) || y < 0 || y >= this->in_height)
            valid = false;

        this->map_cache[i] = std::make_tuple(x, y, valid);
    }

}

std::pair<double, double> Remapper::rotate(double lon, double lat) {
    double x = cos(lon) * cos(lat);
    double z = sin(lon) * cos(lat);
    double y = sin(lat);
    double xx, yy, zz;

    #define R(N) ((N)/180.0*M_PI)
    #define C(N) cos(R(N))
    #define S(N) sin(R(N))
    // rotate base on Z-axis
    xx = x * C(rotate_z) - y * S(rotate_z);
    yy = x * S(rotate_z) + y * C(rotate_z);
    x = xx; y = yy;
    // rotate base on Y-axis
    xx = x * C(rotate_y) + z * S(rotate_y);
    zz = -x * S(rotate_y) + z * C(rotate_y);
    x = xx; z = zz;
    // rotate base on X-axis
    yy = y * C(rotate_x) - z * S(rotate_x);
    zz = y * S(rotate_x) + z * C(rotate_x);
    y = yy; z = zz;
    #undef R
    #undef C
    #undef S

    return std::make_pair<double, double>(atan2(z, x), asin(y));
}
