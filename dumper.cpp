/* 
* @Author: BlahGeek
* @Date:   2015-11-09
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-09
*/

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include "./libmap.hpp"

using namespace vr;

int main(int argc, char * const argv[]){

    const char * usage = "Usage: %s [OPTIONS] -o OUTPUT_FILE CONFIG_JSON\n"
                         "Options:\n"
                         "    -w X:    Set output width, default to 3840\n"
                         "    -h X:    Set output height, default to 0 (do not set width and height both)\n"
                         "";

    int opt_width = 3840;
    int opt_height = 0;
    char * opt_outfile = NULL;

    int opt_ret;
    while((opt_ret = getopt(argc, argv, "w:h:o:")) != -1) {
        switch(opt_ret) {
            case 'w': opt_width = atoi(optarg); break;
            case 'h': opt_height = atoi(optarg); break;
            case 'o': opt_outfile = optarg; break;
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

    auto remapper = MultiMapper::New(options["output"]["type"], options["output"]["options"],
                                     opt_width, opt_height);

    auto out_size = remapper->get_output_size();
    fprintf(stderr, "Output: %dx%d %s\n", out_size.width, out_size.height, 
            options["output"]["type"].get<std::string>().c_str());

    for(auto i: options["inputs"]) {
        int width = i["width"].get<int>();
        int height = i["height"].get<int>();
        fprintf(stderr, "Input: %dx%d %s\n", width, height, i["type"].get<std::string>().c_str());
        remapper->add_input(i["type"], i["options"], width, height);
    }

    std::ofstream of(opt_outfile);
    remapper->dump(of);
    
    return 0;
}
