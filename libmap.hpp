/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-14
*/

#ifndef VR_LIBMAP_BASE_H
#define VR_LIBMAP_BASE_H value

#include "json.hpp"
#include <utility>
#include <math.h>
#include <assert.h>
#include <exception>
#include <vector>
#include <string>
#include <tuple>

namespace vr {

using json = nlohmann::json;

class NotImplemented: std::exception {};
class OutOfRange: std::exception {};

using PointAndFlag = std::tuple<double, double, bool>;

class Map {
public:
    Map(const json & options) {}

    /**
     * Return aspect ratio of mapped image
     * @return aspect ratio, width / height (usually >= 1.0)
     */
    virtual double get_aspect_ratio() {
        return 1.0;
    }

protected:
    /**
     * Map sphere coordinate to image
     * @param lon [-PI, +PI]
     * @param lat [-PI/2, +PI/2]
     * @return x, y in [0, 1), may raise OutOfRange
     */
    virtual std::pair<double, double> lonlat_to_xy(double lon, double lat) {
        throw NotImplemented();
    }

    /**
     * Map image coordinate to sphere
     * may raise std::string on error
     */
    virtual std::pair<double, double> xy_to_lonlat(double x, double y) {
        throw NotImplemented();
    }

public:
    /**
     * @return tuple of (x, y, valid)
     */
    virtual std::vector<PointAndFlag> 
        lonlat_to_xy_batch(const std::vector<std::pair<double, double>> & points);

    virtual std::vector<PointAndFlag>
        xy_to_lonlat_batch(const std::vector<std::pair<double, double>> & points);
};

std::unique_ptr<Map> NewMap(const std::string & type, const json & options);

class Remapper {
private:
    std::unique_ptr<Map> in_map, out_map;
    double rotate_z, rotate_y, rotate_x;

    int in_width, in_height;
    int out_width, out_height;

    std::vector<PointAndFlag> map_cache;

    std::pair<double, double> rotate(double lon, double lat);
public:
    /**
     * @param rotate in degree
     */
    Remapper(const std::string & from, const json & from_opts, 
             const std::string & to, const json & to_opts,
             double rotate_z, double rotate_y, double rotate_x,
             int in_width, int in_height, int out_width, int out_height);
    std::pair<int, int> get_output_size() {
        return std::make_pair(this->out_width, this->out_height);
    }
    PointAndFlag get_map(int w, int h) {
        assert(w >= 0 && w < out_width && h >= 0 && h < out_height);
        int index = h * out_width + w;
        return this->map_cache[index];
    }
};

}

#endif
