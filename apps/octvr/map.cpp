/* 
* @Author: BlahGeek
* @Date:   2015-11-09
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-21
*/

#include <iostream>
#include <fstream>
#include <thread>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/cuda.hpp"

#include "cvconfig.h"
#include "./octvr.hpp"

#include <utility>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#elif defined(_WIN32)
#include "getopt.h"
#endif

using namespace vr;

std::pair<std::vector<cv::UMat>, std::vector<cv::Size>> readImages(std::vector<std::string> filenames) {
    std::vector<cv::UMat> imgs;
    std::vector<cv::Size> sizes;
    for(size_t i = 0 ; i < filenames.size() ; i += 1) {
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

void cvtBGRtoNV12(const cv::UMat & src, cv::UMat & dst) {
    cv::Mat yuv;
    cv::cvtColor(src, yuv, cv::COLOR_BGR2YUV);
    CV_Assert(yuv.size() == src.size());
    CV_Assert(yuv.type() == CV_8UC3);

    dst.create(src.rows + src.rows / 2, src.cols, CV_8U);
    cv::Mat dst_m = dst.getMat(cv::ACCESS_WRITE);

    for(int h = 0 ; h < src.rows ; h += 1) {
        uint8_t * dst_row = dst_m.ptr<uint8_t>(h);
        for(int w = 0 ; w < src.cols ; w += 1) {
            dst_row[w] = yuv.at<uint8_t>(h, w*3);
        }
    }
    for(int h = 0 ; h < src.rows / 2 ; h += 1) {
        uint8_t * dst_row = dst_m.ptr<uint8_t>(h + src.rows);
        for(int w = 0 ; w < src.cols ; w += 1) {
            if(w % 2 == 0)
                dst_row[w] = yuv.at<uint8_t>(h*2, w*3+2);
            else
                dst_row[w] = yuv.at<uint8_t>(h*2, w*3+1);
        }
    }
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
    auto remapper = new FastMapper(map_template, in_sizes);
#endif

    auto output_size = map_template.out_size;
    std::cerr << "Done. Output size = " << output_size << std::endl;

#ifdef HAVE_CUDA
    std::vector<cv::Mat> ms;
    for(auto & um: imgs) {
        cv::Mat x;
        um.copyTo(x);
        ms.push_back(x);
    }
    cv::Mat output(output_size, CV_8UC3);
    cv::Mat output2(output_size, CV_8UC3);
    cv::Mat output3(output_size, CV_8UC3);

    async_remapper->push(ms, output);
    async_remapper->push(ms, output2);
    async_remapper->push(ms, output3);
    async_remapper->pop();
    async_remapper->pop();
    async_remapper->pop();
#else
    cv::UMat output(output_size, CV_8UC3);

    std::vector<cv::UMat> imgs_nv12(imgs.size());
    for(size_t i = 0 ; i < imgs.size() ; i += 1) {
        cvtBGRtoNV12(imgs[i], imgs_nv12[i]);
        cv::cvtColor(imgs_nv12[i], imgs[i], cv::COLOR_YUV2BGR_NV12);
    }
    cv::UMat output_nv12;
    // remapper->stitch(imgs, output);
    remapper->stitch_nv12(imgs_nv12, output_nv12);
    cv::cvtColor(output_nv12, output, cv::COLOR_YUV2BGR_NV12);
#endif

    cv::imwrite(output_filename, output);

    return 0;
}
