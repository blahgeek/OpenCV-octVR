/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-28
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

/**
 * Stitch multiple images into single one
 */
class MultiMapper {
public:
    // Construct MultiMapper using json model.
    static MultiMapper *
        New(const std::string & to, const json & to_opts, 
            int out_width, int out_height);
    // Construct MultiMapper using dumped data file (using dump())
    static MultiMapper * New(std::ifstream & f);

    virtual void add_input(const std::string & from, const json & from_opts) = 0;

    // call be after all add_input()
    // or constructing with data file
    virtual void prepare() = 0;

    virtual cv::Size get_output_size() = 0;
    /**
     * Generate output image
     * @param  inputs Input images, in BGR (CV_8UC3)
     */
    virtual void get_output(const std::vector<cv::cuda::HostMem> & inputs, cv::Mat & output) = 0;

    /**
     * Generate single output image, call this if and if only there's only one input
     * @param input  Input image, in BGR
     * @param output Output image, in BGR
     */
    virtual void get_single_output(const cv::Mat & input, cv::Mat & output) = 0;

    /**
     * Normally, compensator gain will be computed based on the first call of get_output
     * Call reset_compensator if you want to reset(re-compute) it
     */
    virtual void reset_compensator() = 0;

    virtual void dump(std::ofstream & f) = 0;
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
