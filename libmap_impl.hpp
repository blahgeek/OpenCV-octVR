/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-07
*/

#ifndef VR_LIBMAP_IMPL_H
#define VR_LIBMAP_IMPL_H value

#include "./camera.hpp"
#include "./libmap.hpp"
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include <opencv2/core/cuda.hpp>

#ifdef WITH_DONGLE_LICENSE
#include "dongle_license.h"
#endif

namespace vr {

using cv::cuda::GpuMat;
using cv::UMat;

class Mapper {

#ifdef WITH_DONGLE_LICENSE
private:
    license_t lic_t;
    int lic_cnt;
#endif

private:
    cv::Size out_size;

    std::vector<UMat> map1s;
    std::vector<UMat> map2s;
    std::vector<UMat> masks;
    std::vector<UMat> feather_masks;

public:
    Mapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes);
    void stitch(const std::vector<UMat> & inputs, UMat & output);
    // void remap(const std::vector<GpuMat> & inputs, GpuMat & output);
};

}

#endif
