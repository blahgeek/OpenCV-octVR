/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-12
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
#include <opencv2/core/core.hpp>

namespace vr {

using json = nlohmann::json;

class MultiMapper {
public:
    // Construct MultiMapper using json model.
    static MultiMapper *
        New(const std::string & to, const json & to_opts, 
            int out_width, int out_height);
    // Construct MultiMapper using dumped data file (using dump())
    static MultiMapper * New(std::ifstream & f);

    virtual void add_input(const std::string & from, const json & from_opts,
                           int in_width, int in_height) = 0;

    virtual cv::Size get_output_size() = 0;
    virtual cv::Size get_input_size(int index) = 0;
    /**
     * Generate output image
     * @param  inputs Input images, in BGR (CV_8UC3)
     */
    virtual void get_output(const std::vector<cv::UMat> & inputs, cv::UMat & output) = 0;

    virtual void dump(std::ofstream & f) = 0;
};

}

#endif
