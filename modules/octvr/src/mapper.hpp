/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-19
*/

#ifndef VR_LIBMAP_IMPL_H
#define VR_LIBMAP_IMPL_H value

#include "octvr.hpp"
#include "./camera.hpp"
#include "cvconfig.h"

#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/blenders.hpp"

#include "opencv2/core/cuda.hpp"

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
    std::unique_ptr<cv::detail::GainCompensatorGPU> compensator;
    std::unique_ptr<cv::detail::GPUStaticBlender> blender;

    double working_scale;

private:
    std::vector<cv::cuda::Stream> streams;
    std::vector<GpuMat> rgb_inputs;
    std::vector<GpuMat> rgba_inputs;
    std::vector<GpuMat> warped_imgs;
    std::vector<GpuMat> warped_imgs_scale;
    GpuMat result;

public:
    // blend: 0  : Do not blend
    //        > 0: Multi-band blend width
    //        < 0: Feather blend width
    Mapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes, 
           int blend=128, bool enable_gain_compensator=true);
    void stitch(std::vector<GpuMat> & inputs, GpuMat & output);
};

}

#endif
