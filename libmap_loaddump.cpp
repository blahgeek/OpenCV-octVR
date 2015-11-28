/* 
* @Author: BlahGeek
* @Date:   2015-11-09
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-29
*/

#include <iostream>
#include <string.h>
#include "./libmap_impl.hpp"

using namespace vr;

static const char * DUMP_MAGIC = "octVR";

void MultiMapperImpl::dump(std::ofstream & f) {
    f.write(DUMP_MAGIC, strlen(DUMP_MAGIC));

    auto W64i = [&](int64_t x) {
        f.write(reinterpret_cast<char *>(&x), sizeof(int64_t));
    };
    auto Wd = [&](double x) {
        f.write(reinterpret_cast<char *>(&x), sizeof(double));
    };

    W64i(out_size.width);
    W64i(out_size.height);

    W64i(map1s.size());
    for(int i = 0 ; i < map1s.size() ; i += 1) {
        W64i(0);
        W64i(0); // OMG backwards compatible
        assert(map1s[i].type() == CV_32FC1 && map1s[i].size() == out_size);
        assert(map2s[i].type() == CV_32FC1 && map2s[i].size() == out_size);
        assert(masks[i].type() == CV_8UC1 && masks[i].size() == out_size);

        cv::Mat map1, map2, mask;
        map1s[i].download(map1);
        map2s[i].download(map2);
        masks[i].download(mask);

        for(int k = 0 ; k < out_size.height ; k += 1) {
            f.write(map1.ptr<char>(k), out_size.width * 4); // CV_16SC2
            f.write(map2.ptr<char>(k), out_size.width * 4); // CV_16UC1
            f.write(mask.ptr<char>(k), out_size.width); // CV_8UC1
        }

        Wd(working_scales[i]);
        W64i(scaled_masks[i].cols);
        W64i(scaled_masks[i].rows);
        cv::Mat scaled_mask;
        scaled_masks[i].download(scaled_mask);

        assert(scaled_mask.type() == CV_8UC1);
        for(int k = 0 ; k < scaled_mask.rows ; k += 1)
            f.write(scaled_mask.ptr<char>(k), scaled_mask.cols);
    }

}

MultiMapperImpl::MultiMapperImpl(std::ifstream & f) {
    char read_magic[16];
    f.read(read_magic, strlen(DUMP_MAGIC));
    if(strncmp(read_magic, DUMP_MAGIC, strlen(DUMP_MAGIC)) != 0)
        throw std::string("Invalid data file");

    auto R64i = [&]() -> int64_t {
        int64_t _ret = 0;
        f.read(reinterpret_cast<char *>(&_ret), sizeof(int64_t));
        return _ret;
    };
    auto Rd = [&]() -> double {
        double _ret = 0;
        f.read(reinterpret_cast<char *>(&_ret), sizeof(double));
        return _ret;
    };

    this->out_size.width = R64i();
    this->out_size.height = R64i();

    int in_count = R64i();
    for(int i = 0 ; i < in_count ; i += 1) {
        R64i(); R64i(); // OMG, backward compatible

        cv::Mat map1(out_size, CV_32FC1);
        cv::Mat map2(out_size, CV_32FC1);
        cv::Mat mask(out_size, CV_8UC1);
        for(int k = 0 ; k < out_size.height ; k += 1) {
            f.read(map1.ptr<char>(k), out_size.width * 4);
            f.read(map2.ptr<char>(k), out_size.width * 4);
            f.read(mask.ptr<char>(k), out_size.width);
        }
        GpuMat map1_gpu, map2_gpu, mask_gpu;
        map1_gpu.upload(map1);
        map2_gpu.upload(map2);
        mask_gpu.upload(mask);
        this->map1s.push_back(map1_gpu);
        this->map2s.push_back(map2_gpu);
        this->masks.push_back(mask_gpu);

        working_scales.push_back(Rd());
        auto scaled_width = R64i();
        auto scaled_height = R64i();
        cv::Mat scaled_mask(cv::Size(scaled_width, scaled_height), CV_8UC1);
        for(int k = 0 ; k < scaled_height ; k += 1)
            f.read(scaled_mask.ptr<char>(k), scaled_width);
        GpuMat scaled_mask_u;
        scaled_mask_u.upload(scaled_mask);
        scaled_masks.push_back(scaled_mask_u);
    }
}
