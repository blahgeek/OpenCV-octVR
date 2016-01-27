/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-27
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
#include <stdint.h>
#include <iostream>
#include <sys/time.h>

#include "opencv2/core.hpp"

namespace vr {

using json = nlohmann::json;

// Multiple input -> single output
class MapperTemplate {
public:
    std::string out_type;
    json out_opts;
    cv::Size out_size;

    typedef struct {
        cv::Mat map1, map2;
        cv::Mat mask;
    } Input;

    std::vector<Input> inputs;
    std::vector<Input> overlay_inputs;

    std::vector<cv::Mat> seam_masks;  // only for inputs (not overlay_inputs)

public:
    // Create new template
    // width/height must be suitable to output model
    MapperTemplate(const std::string & to,
                   const json & to_opts,
                   int width, int height);
    void add_input(const std::string & from, const json & from_opts, bool overlay=false);
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

class FastMapper {

private:
    std::vector<cv::Size> in_sizes;
    cv::Size out_size;

    std::vector<cv::UMat> map1s, half_map1s;
    std::vector<cv::UMat> map2s, half_map2s;
    std::vector<cv::UMat> masks, half_masks;
    std::vector<cv::UMat> feather_masks, half_feather_masks;

private:
    cv::UMat output_f_c0;
    std::vector<cv::UMat> output_f_c1c2;
    std::vector<cv::UMat> input_c1c2, remapped_channels;
    cv::UMat output_c1c2_merge;

public:
    FastMapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes);
    void stitch(const std::vector<cv::UMat> & inputs, cv::UMat & output);
    void stitch_nv12(const std::vector<cv::UMat> & inputs, cv::UMat & output);
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
