/* 
* @Author: BlahGeek
* @Date:   2016-02-28
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-29
*/

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

#include "opencv2/stitching/detail/exposure_compensate.hpp"

#include "octvr.hpp"

using namespace vr;

MapperTemplate read_template(const char * filename) {
    std::cerr << "Reading template " << filename << std::endl;
    std::ifstream f(filename, std::ios::binary);
    MapperTemplate ret(f);
    f.close();
    return ret;
}

cv::UMat read_image(const char * filename) {
    std::cerr << "Reading image " << filename << std::endl;
    cv::UMat ret;
    cv::Mat img = cv::imread(filename);
    img.copyTo(ret);
    return ret;
}

void save_images(const std::vector<cv::UMat> & images, std::string name) {
    for(size_t i = 0 ; i < images.size() ; i += 1) {
        char tmp[128];
        snprintf(tmp, 128, "__%s_%ld.bmp", name.c_str(), i);
        std::cerr << "Saving image " << tmp << std::endl;
        cv::imwrite(tmp, images[i]);
    }
}

int main(int argc, char const *argv[]) {
    const char * usage = "%s MAP.dat MAP_R0.dat ... input0.bmp ... output0.bmp ...";
    if(argc < 5) {
        printf(usage, argv[0]);
        return 1;
    }

    auto mt = read_template(argv[1]);
    
    argc -= 2;
    argv += 2;
    int num_images = argc / 3;
    std::cerr << "Processing " << num_images << "images" << std::endl;

    CV_Assert(mt.inputs.size() == num_images);

    std::vector<MapperTemplate> mt_r;
    for(int i = 0 ; i < num_images ; i += 1)
        mt_r.push_back(read_template(argv[i]));

    argv += num_images;

    std::vector<cv::UMat> inputs;
    for(int i = 0 ; i < num_images ; i += 1)
        inputs.push_back(read_image(argv[i]));

    argv += num_images;

    std::vector<cv::UMat> map1s, map2s, masks;
    for(int i = 0 ; i < num_images ; i += 1) {
        auto & in = mt.inputs[i];
        cv::UMat map1, map2;
        cv::convertMaps(in.map1 * inputs[i].cols, 
                        in.map2 * inputs[i].rows, 
                        map1, map2, CV_16SC2);
        map1s.push_back(map1);
        map2s.push_back(map2);

        cv::UMat mask;
        in.mask.copyTo(mask);
        masks.push_back(mask);
    }

    std::vector<cv::UMat> map1s_r, map2s_r;
    for(int i = 0 ; i < num_images ; i += 1) {
        CV_Assert(mt_r[i].inputs.size() == 1);
        cv::UMat map1, map2;
        cv::convertMaps(mt_r[i].inputs[0].map1 * mt.out_size.width,
                        mt_r[i].inputs[0].map2 * mt.out_size.height,
                        map1, map2, CV_16SC2);
        map1s_r.push_back(map1);
        map2s_r.push_back(map2);
    }

    std::vector<cv::UMat> remapped_inputs(num_images);
    for(int i = 0 ; i < num_images ; i += 1)
        cv::remap(inputs[i], remapped_inputs[i], 
                  map1s[i], map2s[i], cv::INTER_LINEAR);
    save_images(remapped_inputs, "remapped_inputs");

    cv::detail::BlocksGainCompensator gain_compensator;
    std::vector<std::pair<cv::UMat, uchar> > level_masks;
    for(int i = 0 ; i < num_images ; i += 1)
        level_masks.push_back(std::make_pair(masks[i], 255));
    gain_compensator.feed(std::vector<cv::Point>(num_images, cv::Point(0, 0)),
                          remapped_inputs, level_masks);
    for(int i = 0 ; i < num_images ; i += 1)
        gain_compensator.apply(i, cv::Point(0, 0), remapped_inputs[i], masks[i]);
    save_images(remapped_inputs, "remapped_inputs_gain");

    auto gain_maps = gain_compensator.getGainMaps();
    save_images(gain_maps, "gain_maps");

    std::vector<cv::UMat> gain_maps_remapped(num_images);
    for(int i = 0 ; i < num_images ; i += 1)
        cv::remap(gain_maps[i], gain_maps_remapped[i],
                  map1s_r[i], map2s_r[i], cv::INTER_LINEAR);
    save_images(gain_maps_remapped, "gain_maps_remapped");

    return 0;
}
