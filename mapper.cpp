/* 
* @Author: BlahGeek
* @Date:   2015-11-09
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-16
*/

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/cuda.hpp>
#include "./libmap.hpp"

#include <cuda_profiler_api.h>

using namespace vr;

int main(int argc, char const *argv[]) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " map.dat output.jpg input0.jpg input1.jpg ..." << std::endl
                  << "\tmap.dat can be produced by `dumper`" << std::endl;
        return 1;
    }

    cv::ocl::setUseOpenCL(true);

    char const * map_filename = argv[1];
    char const * output_filename = argv[2];

    std::cerr << "Loading map file " << map_filename << std::endl;
    std::ifstream map_file(map_filename);
    auto remapper = MultiMapper::New(map_file);
    assert(remapper != NULL);
    remapper->prepare();
    auto output_size = remapper->get_output_size();
    std::cerr << "Done. Output size = " << output_size << std::endl;

    //std::vector<cv::Mat> imgs;
    std::vector<cv::cuda::HostMem> imgs;

    for(int i = 3 ; i < argc ; i += 1) {
        char const * img_filename = argv[i];
        std::cerr << "Reading input #" << i-3 << ": " << img_filename << std::endl;
        cv::Mat img = cv::imread(img_filename);
        std::cerr << "Image size = " << img.size() << std::endl;

        assert(img.size() == remapper->get_input_size(i-3));
        cv::cuda::HostMem img_host;
        img.copyTo(img_host);
        imgs.push_back(img_host);
    }

    cv::Mat output(output_size, CV_8UC3);
    std::cerr << "Remapping..." << std::endl;
    cudaProfilerStart();
    remapper->get_output(imgs, output);
    cudaProfilerStop();
    std::cerr << "Done" << std::endl;

    cv::imwrite(output_filename, output);

    return 0;
}
