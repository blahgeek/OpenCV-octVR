/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-20
*/

#ifndef VR_LIBMAP_BASE_H
#define VR_LIBMAP_BASE_H value

#include "json.hpp"
#include <utility>
#include <cmath>
#include <cassert>
#include <exception>
#include <vector>
#include <string>
#include <tuple>
#include <opencv2/core/core.hpp>

namespace vr {

using json = nlohmann::json;

class NotImplemented: std::exception {};

class Map {
protected:
    std::vector<double> rotate_vector;
    cv::Mat rotate_matrix;

protected:
    cv::Point2d sphere_xyz_to_lonlat(const cv::Point3d & xyz);
    cv::Point3d sphere_lonlat_to_xyz(const cv::Point2d & lonlat);

    void sphere_rotate(std::vector<cv::Point3d> & points, bool reverse);

public:
    /**
     * Provide "rotate" optionally
     */
    Map(const json & options);

    /**
     * Return aspect ratio of mapped image
     * @return aspect ratio, width / height (usually >= 1.0)
     */
    virtual double get_aspect_ratio() {
        return 1.0;
    }

protected:
    /**
     * Map object point in sphere, to image point in rectangle
     * @param  lonlat lon in [-PI, +PI), lat in [-PI/2, +PI/2)
     * @return        x, y in [0, 1) or NAN
     */
    virtual cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) {
        throw NotImplemented();
    }

    /**
     * Map image point in rectangle, to object point in sphere
     * @param  xy x, y in [0, 1)
     * @return    lonlat, lon in [-PI, +PI), lat in [-PI/2, +PI/2), may be NAN
     */
    virtual cv::Point2d image_to_obj_single(const cv::Point2d & xy) {
        throw NotImplemented();
    }

public:
    /**
     * Map object points to image points
     */
    virtual std::vector<cv::Point2d> obj_to_image(const std::vector<cv::Point2d> & lonlats);

    /**
     * Map image points to object points
     */
    virtual std::vector<cv::Point2d> image_to_obj(const std::vector<cv::Point2d> & xys);

public:
    static std::unique_ptr<Map> New(const std::string & type, const json & opts);
};

class Remapper {
private:
    std::unique_ptr<Map> in_map, out_map;

    int in_width, in_height;
    int out_width, out_height;

    std::vector<cv::Point2d> map_cache;
public:
    /**
     * @param rotate in degree
     */
    Remapper(const std::string & from, const json & from_opts, 
             const std::string & to, const json & to_opts,
             int in_width, int in_height, int out_width, int out_height);

    cv::Size get_output_size() {
        return cv::Size(this->out_width, this->out_height);
    }

    cv::Point2d get_map(int w, int h) {
        assert(w >= 0 && w < out_width && h >= 0 && h < out_height);
        int index = h * out_width + w;
        return this->map_cache[index];
    }
};

class MultiMapper {
private:
    std::unique_ptr<Map> out_map;
    std::vector<std::unique_ptr<Map>> in_maps;

    cv::Size out_size;
    std::vector<cv::Size> in_sizes;

    std::vector<cv::Point2d> output_map_points;
    std::vector<std::vector<cv::Point2d>> map_caches;

public:
    MultiMapper(const std::string & to, const json & to_opts, 
                int out_width, int out_height);
    void add_input(const std::string & from, const json & from_opts,
                   int in_width, int in_height);

    cv::Size get_output_size() {
        return this->out_size;
    }

    std::pair<int, cv::Point2d> get_map(int w, int h);
};

}

#endif
