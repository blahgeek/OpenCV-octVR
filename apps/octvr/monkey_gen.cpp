/* 
* @Author: BlahGeek
* @Date:   2016-01-02
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-21
*/

#include <iostream>
#include <iostream>
#include <fstream>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <getopt.h>
#elif defined(_WIN32)
#include "getopt.h"
#define snprintf _snprintf
#endif
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/cuda.hpp"
#include "octvr.hpp"
#include <assert.h>

using namespace vr;

int main(int argc, char const *argv[]) {
    if (argc <= 1) {
        fprintf(stderr, "Usage: %s BORDER INPUT OUTPUT\n", argv[0]);
        exit(-1);
    }

    int border = atoi(argv[1]);
    std::ifstream in_f(argv[2]);
    MapperTemplate mt(in_f);

    assert(mt.overlay_inputs.size() == 0);
    std::cout << "Inputs: " << mt.inputs.size() << std::endl;
    std::cout << "Out size: " << mt.out_size << std::endl;

    std::vector<cv::Mat> weight_maps;
    cv::Mat dst_weight_map(mt.out_size, CV_32F);
    dst_weight_map.setTo(1e-5f);

    for(auto & input: mt.inputs) {
        cv::Mat weight_map, tmp;
        cv::distanceTransform(input.mask, weight_map, cv::DIST_L2, 3);
        cv::subtract(weight_map, border, tmp);
        cv::threshold(tmp, weight_map, 0.f, 0.f, cv::THRESH_TOZERO);
        cv::add(weight_map, dst_weight_map, dst_weight_map);
        weight_maps.push_back(weight_map);
    }

    for(int i = 0 ; i < weight_maps.size() ; i += 1) {
        cv::divide(weight_maps[i], dst_weight_map, weight_maps[i]);
        cv::Mat u8;
        weight_maps[i].convertTo(u8, CV_8UC1, 255.0);

        char filename[64];
        snprintf(filename, 64, "%s.%d.png", argv[3], i);
        cv::imwrite(filename, u8);
    }

    return 0;
}
