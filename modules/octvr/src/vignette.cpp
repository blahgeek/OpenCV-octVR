/* 
* @Author: BlahGeek
* @Date:   2016-04-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-22
*/

#if defined( _MSC_VER )
#define _USE_MATH_DEFINES
#endif
#include <iostream>
#include <cmath>

#include "./vignette.hpp"

using namespace vr;

Vignette::Vignette(const rapidjson::Value & options) {
    if(options.HasMember("vignette")) {
        this->valid = true;
        a = options["vignette"][0].GetDouble();
        b = options["vignette"][1].GetDouble();
        c = options["vignette"][2].GetDouble();
        d = options["vignette"][3].GetDouble();
        if(options.HasMember("exposure")) {
            float ev = std::pow(2.0, options["exposure"].GetDouble());
            a /= ev;
            b /= ev;
            c /= ev;
            d /= ev;
        }
        std::cerr << "Vignette params: " << a << ", " << b << ", " << c << ", " << std::endl;
    } else {
        this->valid = false;
        std::cerr << "Vignette not defined" << std::endl;
    }
}

cv::Mat Vignette::getMap(int width, int height) {
    if(!valid)
        return cv::Mat();
    cv::Mat ret(height, width, CV_32F);
    for(int j = 0 ; j < height ; j += 1) {
        float * row = ret.ptr<float>(j);
        for(int i = 0 ; i < width ; i += 1) {
            float r = std::sqrt(std::pow(double(i) / width - 0.5, 2.0) + 
                                std::pow(double(j) / height - 0.5, 2.0));
            row[i] = 1.0 / (a + r * r * (b + r * r * (c + d * r * r)));
        }
    }
    return ret;
}

