/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-01
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

    virtual cv::Size get_output_size() = 0;
    /**
     * Generate output image
     * @param  inputs Input images, in BGR_ (CV_8UC4, last channel is ignored)
     */
    virtual void get_output(const std::vector<cv::cuda::GpuMat> & inputs, cv::cuda::GpuMat & output) = 0;

    /**
     * Generate single output image, call this if and if only there's only one input
     * @param input  Input image, in BGR_
     * @param output Output image, in BGR_
     */
    virtual void get_single_output(const cv::Mat & input, cv::Mat & output) = 0;

    virtual void dump(std::ofstream & f) = 0;
};

class AsyncMultiMapper {
public:
    static AsyncMultiMapper * New(MultiMapper * mapper);

    virtual void push(std::vector<cv::Mat> & inputs,
                      cv::Mat & output) = 0;
    // return the same output as you pushed
    virtual cv::Mat pop() = 0;
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
