/* 
* @Author: BlahGeek
* @Date:   2015-11-09
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-24
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
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/cuda.hpp"
#include "octvr.hpp"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

using namespace vr;

int main(int argc, char * const argv[]){

    const char * usage = "Usage: %s [OPTIONS] -o OUTPUT_FILE CONFIG_JSON\n"
                         "Options:\n"
                         "    -w X:    Set output width, default to 0\n"
                         "    -h X:    Set output height, default to 0\n"
                         "    -d X:    Save masks to debug directory\n"
                         "    -n:      Do not use ROI based stitching\n"
                         "";

    int opt_width = 0;
    int opt_height = 0;
    char * opt_debug = NULL;
    char * opt_outfile = NULL;
    bool opt_roi = true;

    int opt_ret;
    while((opt_ret = getopt(argc, argv, "w:h:o:d:n")) != -1) {
        switch(opt_ret) {
            case 'w': opt_width = atoi(optarg); break;
            case 'h': opt_height = atoi(optarg); break;
            case 'o': opt_outfile = optarg; break;
            case 'd': opt_debug = optarg; break;
            case 'n': opt_roi = false; break;
            default:
                fprintf(stderr, usage, argv[0]);
                return 0;
        }
    }

    argc -= optind;
    if(argc == 0 || opt_outfile == NULL) {
        fprintf(stderr, usage, argv[0]);
        return 0;
    }
    argv += optind;

    rapidjson::Document options;
    std::ifstream f(argv[0]);
    rapidjson::IStreamWrapper ifs(f);
    options.ParseStream(ifs);

    MapperTemplate mt(options["output"]["type"].GetString(),
                      options["output"]["options"],
                      opt_width, opt_height);

    auto out_size = mt.out_size;
    fprintf(stderr, "Output: %dx%d %s\n", out_size.width, out_size.height, 
            options["output"]["type"].GetString());

    for(auto i = options["inputs"].Begin() ; i != options["inputs"].End() ; i ++ ) {
        fprintf(stderr, "Input: %s\n", (*i)["type"].GetString());
        mt.add_input((*i)["type"].GetString(), (*i)["options"], false, opt_roi);
    }
    if(options.HasMember("overlays")) {
        for(auto i = options["overlays"].Begin() ; i != options["overlays"].End() ; i ++) {
            fprintf(stderr, "Overlay input: %s\n", (*i)["type"].GetString());
            mt.add_input((*i)["type"].GetString(), (*i)["options"], true, opt_roi);
        }
    }
    
    if(!(argc == 1 || argc - 1 == options["inputs"].Size())) {
        fprintf(stderr, "Invalid argument\n");
        return 1;
    }
    if(argc > 1) {
        std::vector<cv::Mat> imgs;
        for(int i = 1 ; i < argc ; i += 1) {
            fprintf(stderr, "Reading image %s...\n", argv[i]);
            cv::Mat img = cv::imread(argv[i], 1);
            imgs.push_back(img);
        }
        mt.create_masks(imgs);
    }

    std::ofstream of(opt_outfile, std::ios::binary);
    mt.dump(of);

#define SAVE_MAT(D, N, S, MAT) \
    do { \
        char tmp[64]; \
        snprintf(tmp, 64, "%s/%s_%d.jpg", D, S, N);\
        std::cerr << "Saving " << tmp << std::endl;\
        cv::imwrite(tmp, MAT); \
    } while(false)

#define SAVE_MAT_VEC(D, S, MATS) \
    do { \
        for(int __i = 0 ; __i < MATS.size() ; __i += 1) \
            SAVE_MAT(D, __i, S, MATS[__i]); \
    } while(false)

    if(opt_debug) {
        for(int i = 0 ; i < mt.inputs.size() ; i += 1) {
            SAVE_MAT(opt_debug, i, "masks", mt.inputs[i].mask);
            SAVE_MAT(opt_debug, i, "seam_masks", mt.seam_masks[i]);
        }
    }
    // remapper->prepare(std::vector<cv::Size>(options["inputs"].size(), cv::Size(1,1)));
    // remapper->debug_save_mats();
    
    return 0;
}
