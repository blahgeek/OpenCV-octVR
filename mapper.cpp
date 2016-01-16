/* 
* @Author: BlahGeek
* @Date:   2015-11-09
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-16
*/

#include <iostream>
#include <fstream>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/cuda.hpp>
#include "./libmap.hpp"
#include "./libmap_impl.hpp"
#include <utility>
#include <unistd.h>

#ifdef HAVE_CUDA
#include <cuda_profiler_api.h>
#endif

using namespace vr;

std::pair<std::vector<cv::UMat>, std::vector<cv::Size>> readImages(std::vector<std::string> filenames) {
    std::vector<cv::UMat> imgs;
    std::vector<cv::Size> sizes;
    for(int i = 0 ; i < filenames.size() ; i += 1) {
        std::cerr << "Reading input #" << i << ": " << filenames[i] << std::endl;

        cv::Mat img = cv::imread(filenames[i]);
        std::cerr << "Image size = " << img.size() << std::endl;
        sizes.push_back(img.size());

        cv::UMat uimg;
        img.copyTo(uimg);
        imgs.push_back(uimg);
    }
    return std::make_pair(imgs, sizes);
}

int main(int argc, char const *argv[]) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " map.dat output.jpg input0.jpg input1.jpg ..." << std::endl
                  << "\tmap.dat can be produced by `dumper`" << std::endl;
        return 1;
    }

    char const * map_filename = argv[1];
    char const * output_filename = argv[2];
    std::vector<std::string> input_filenames(argv+3, argv+argc);
    std::vector<cv::UMat> imgs;
    std::vector<cv::Size> in_sizes;
    std::tie(imgs, in_sizes) = readImages(input_filenames);

    std::cerr << "Loading map file " << map_filename << std::endl;
    std::ifstream map_file(map_filename);

    MapperTemplate map_template(map_file);
#ifdef HAVE_CUDA
    auto async_remapper = AsyncMultiMapper::New(map_template, in_sizes);
    assert(async_remapper != NULL);
#else
    auto remapper = new CPUMapper(map_template, in_sizes);
#endif

    auto output_size = map_template.out_size;
    std::cerr << "Done. Output size = " << output_size << std::endl;

#ifdef HAVE_CUDA
    cv::Mat output(output_size, CV_8UC3);
    cv::Mat output2(output_size, CV_8UC3);
    cv::Mat output3(output_size, CV_8UC3);

    cudaProfilerStart();
    async_remapper->push(imgs, output);
    async_remapper->push(imgs, output2);
    async_remapper->push(imgs, output3);
    async_remapper->pop();
    async_remapper->pop();
    async_remapper->pop();

    cudaProfilerStop();
#else
    cv::UMat output(output_size, CV_8UC3);
    remapper->stitch(imgs, output);
#endif

    cv::imwrite(output_filename, output);

    return 0;
}
