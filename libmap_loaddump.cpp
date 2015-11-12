/* 
* @Author: BlahGeek
* @Date:   2015-11-09
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-12
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

    W64i(out_size.width);
    W64i(out_size.height);

    W64i(in_sizes.size());
    for(int i = 0 ; i < in_sizes.size() ; i += 1) {
        W64i(in_sizes[i].width);
        W64i(in_sizes[i].height);
        assert(map1s[i].type() == CV_16SC2 && map1s[i].size() == out_size);
        assert(map2s[i].type() == CV_16UC1 && map2s[i].size() == out_size);
        assert(masks[i].type() == CV_8UC1 && masks[i].size() == out_size);

        auto map1 = map1s[i].getMat(cv::ACCESS_READ);
        auto map2 = map2s[i].getMat(cv::ACCESS_READ);
        auto mask = masks[i].getMat(cv::ACCESS_READ);

        for(int k = 0 ; k < out_size.height ; k += 1) {
            f.write(map1.ptr<char>(k), out_size.width * 4); // CV_16SC2
            f.write(map2.ptr<char>(k), out_size.width * 2); // CV_16UC1
            f.write(mask.ptr<char>(k), out_size.width); // CV_8UC1
        }
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

    this->out_size.width = R64i();
    this->out_size.height = R64i();

    int in_count = R64i();
    for(int i = 0 ; i < in_count ; i += 1) {
        int width = R64i();
        int height = R64i();
        this->in_sizes.push_back(cv::Size(width, height));

        cv::Mat map1(out_size, CV_16SC2);
        cv::Mat map2(out_size, CV_16UC1);
        cv::Mat mask(out_size, CV_8UC1);
        for(int k = 0 ; k < out_size.height ; k += 1) {
            f.read(map1.ptr<char>(k), out_size.width * 4);
            f.read(map2.ptr<char>(k), out_size.width * 2);
            f.read(mask.ptr<char>(k), out_size.width);
        }
        cv::UMat map1u, map2u, masku;
        map1.copyTo(map1u);
        map2.copyTo(map2u);
        mask.copyTo(masku);
        this->map1s.push_back(map1u);
        this->map2s.push_back(map2u);
        this->masks.push_back(masku);
    }
}
