/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-14
*/

#include <iostream>
#include "./libmap.hpp"

#include "./equirectangular.hpp"
#include "./normal.hpp"
#include "./pinhole_cam.hpp"

using namespace vr;

std::unique_ptr<Map> vr::NewMap(const std::string & type, const json & options) {
    #define X(s, t) \
        else if (type == s) return std::unique_ptr<Map>(new t(options));

    if(false){}

    X("normal", Normal)
    X("pinhole", PinholeCamera)
    X("equirectangular", Equirectangular)

    return nullptr;

    #undef X
}

Remapper::Remapper(const std::string & from, const json & from_opts,
                   const std::string & to, const json & to_opts,
                   double rotate_z, double rotate_y, double rotate_x,
                   int in_width, int in_height, int out_width, int out_height):
rotate_z(rotate_z), rotate_y(rotate_y), rotate_x(rotate_x),
in_width(in_width), in_height(in_height), out_width(out_width), out_height(out_height) {
    this->in_map = NewMap(from, from_opts);
    this->out_map = NewMap(to, to_opts);
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

    int pixel_count = this->out_height * this->out_width;
    this->map_cache.resize(pixel_count);
    this->map_valid.resize(pixel_count);

    for(int j = 0 ; j < this->out_height ; j += 1) {
        for(int i = 0 ; i < this->out_width ; i += 1) {
            int index = j * this->out_width + i;
            this->map_valid[index] = true;

            double x = double(i) / this->out_width;
            double y = double(j) / this->out_height;

            try {
                auto lonlat = out_map->xy_to_lonlat(x, y);
                lonlat = this->rotate(lonlat.first, lonlat.second);
                auto dst_xy = in_map->lonlat_to_xy(lonlat.first, lonlat.second);
                this->map_cache[index].first = int(dst_xy.first * this->in_width);
                this->map_cache[index].second = int(dst_xy.second * this->in_height);

                if(this->map_cache[index].first < 0 || 
                   this->map_cache[index].first >= this->in_width ||
                   this->map_cache[index].second < 0 || 
                   this->map_cache[index].second >= this->in_height)
                    throw OutOfRange();

            } catch(OutOfRange & e) {
                this->map_valid[index] = false;
            }
        }
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
