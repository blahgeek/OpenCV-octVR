/* 
* @Author: BlahGeek
* @Date:   2016-04-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-22
*/

#ifndef VR_VIGNETTE_H_
#define VR_VIGNETTE_H_ value

#include "opencv2/core.hpp"
#include "rapidjson/document.h"

namespace vr {

class Vignette {

private:
    bool valid;
    float a, b, c, d;

public:
    Vignette(const rapidjson::Value & options);

    cv::Mat getMap(int width, int height);
};

}

#endif
