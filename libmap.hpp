/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-07
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
#include <fstream>
#include <opencv2/core.hpp>
#include <stdint.h>
#include <iostream>
#include <sys/time.h>

namespace vr {

using json = nlohmann::json;

// Multiple input -> single output
class MapperTemplate {
public:
    std::string out_type;
    json out_opts;
    cv::Size out_size;

    std::vector<cv::Mat> map1s;
    std::vector<cv::Mat> map2s;
    std::vector<cv::Mat> masks;
    std::vector<cv::Mat> seam_masks;

public:
    // Create new template
    // width/height must be suitable to output model
    MapperTemplate(const std::string & to,
                   const json & to_opts,
                   int width, int height);
    void add_input(const std::string & from, const json & from_opts);
    // Prepare seam masks with provided images (optional)
    void create_masks(const std::vector<cv::Mat> & imgs = std::vector<cv::Mat>());
    void dump(std::ofstream & f);

    // Load existing template
    explicit MapperTemplate(std::ifstream & f);
};

class AsyncMultiMapper {
public:
    static AsyncMultiMapper * New(const std::vector<MapperTemplate> & mts, std::vector<cv::Size> in_sizes, int blend=128);
    static AsyncMultiMapper * New(const MapperTemplate & mt, std::vector<cv::Size> in_sizes, int blend=128);

    /**
     * Push one frame
     * @param inputs Input images, in RGB
     * @param output Output images, in RGB
     */
    virtual void push(std::vector<cv::Mat> & inputs,
                      std::vector<cv::Mat> & outputs) = 0;
    // Single output
    virtual void push(std::vector<cv::Mat> & inputs, cv::Mat & outputs) = 0;
    virtual void pop() = 0;

    virtual ~AsyncMultiMapper() {}
};

class Timer {
protected:
    int64_t t;
    std::string name;

public:
    explicit Timer(std::string name);
    Timer();
    void tick(std::string msg);
};

}

#endif
