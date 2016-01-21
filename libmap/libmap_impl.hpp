/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-16
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

    int nonoverlay_num;
    std::vector<GpuMat> map1s; // CV_32FC1
    std::vector<GpuMat> map2s;
    std::vector<GpuMat> masks; 
    std::vector<GpuMat> seam_masks;

private:
    cv::Ptr<cv::detail::GainCompensatorGPU> compensator;
    cv::Ptr<cv::detail::GPUStaticBlender> blender;

    double working_scale;

private:
    std::vector<cv::cuda::Stream> streams;
    std::vector<GpuMat> gpu_inputs;
    std::vector<GpuMat> warped_imgs;
    std::vector<GpuMat> warped_imgs_scale;
    GpuMat result;

public:
    // blend: 0  : Do not blend
    //        > 0: Multi-band blend width
    //        < 0: Feather blend width
    Mapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes, int blend=128);
    void stitch(const std::vector<GpuMat> & inputs, GpuMat & output);
    void remap(const std::vector<GpuMat> & inputs, GpuMat & output);
};

class CPUMapper {

private:
    cv::Size out_size;

    std::vector<UMat> map1s;
    std::vector<UMat> map2s;
    std::vector<UMat> masks;
    std::vector<UMat> feather_masks;

public:
    CPUMapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes);
    void stitch(const std::vector<UMat> & inputs, UMat & output);
};

}

#endif
