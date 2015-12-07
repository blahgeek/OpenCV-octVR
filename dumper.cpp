/* 
* @Author: BlahGeek
* @Date:   2015-11-09
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-07
*/

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/cuda.hpp>
#include "./libmap.hpp"

using namespace vr;

int main(int argc, char * const argv[]){

    const char * usage = "Usage: %s [OPTIONS] -o OUTPUT_FILE CONFIG_JSON\n"
                         "Options:\n"
                         "    -w X:    Set output width, default to 3840\n"
                         "    -h X:    Set output height, default to 0 (do not set width and height both)\n"
                         "    -d X:    Save masks to debug directory\n"
                         "";

    int opt_width = 3840;
    int opt_height = 0;
    char * opt_debug = NULL;
    char * opt_outfile = NULL;

    int opt_ret;
    while((opt_ret = getopt(argc, argv, "w:h:o:d:")) != -1) {
        switch(opt_ret) {
            case 'w': opt_width = atoi(optarg); break;
            case 'h': opt_height = atoi(optarg); break;
            case 'o': opt_outfile = optarg; break;
            case 'd': opt_debug = optarg; break;
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

    json options;
    std::ifstream f(argv[0]);
    options << f;

    MapperTemplate mt(options["output"]["type"],
                      options["output"]["options"],
                      opt_width, opt_height);

    auto out_size = mt.out_size;
    fprintf(stderr, "Output: %dx%d %s\n", out_size.width, out_size.height, 
            options["output"]["type"].get<std::string>().c_str());

    for(auto i: options["inputs"]) {
        fprintf(stderr, "Input: %s\n", i["type"].get<std::string>().c_str());
        mt.add_input(i["type"], i["options"]);
    }

    std::ofstream of(opt_outfile);
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
        SAVE_MAT_VEC(opt_debug, "masks", mt.masks);
        SAVE_MAT_VEC(opt_debug, "seam_masks", mt.seam_masks);
    }
    // remapper->prepare(std::vector<cv::Size>(options["inputs"].size(), cv::Size(1,1)));
    // remapper->debug_save_mats();
    
    return 0;
}
