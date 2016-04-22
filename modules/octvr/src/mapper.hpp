/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-22
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
    bool with_logo = true;

private:
    cv::Size stitch_size, scaled_output_size;

    int nonoverlay_num;
    std::vector<GpuMat> map1s; // CV_32FC1
    std::vector<GpuMat> map2s;
    std::vector<GpuMat> masks; 
    std::vector<GpuMat> vignette_maps;
    std::vector<GpuMat> seam_masks;
    std::vector<cv::Rect> rois;

#ifdef WITH_OCTVR_LOGO
    GpuMat logo_data, logo_mask;
#endif

private:
    std::unique_ptr<cv::detail::GainCompensatorGPU> compensator;
    std::unique_ptr<cv::detail::GPUStaticBlender> blender;

    double working_scale;
    std::vector<cv::Rect> working_scaled_rois;

private:
    std::vector<cv::cuda::Stream> streams;
    std::vector<GpuMat> rgb_inputs;
    std::vector<GpuMat> rgba_inputs;
    std::vector<GpuMat> warped_imgs;
    std::vector<GpuMat> warped_imgs_scale;
    std::vector<GpuMat> warped_imgs_rgb;
    GpuMat result, result_scaled;

public:
    // blend: 0  : Do not blend
    //        > 0: Multi-band blend width
    //        < 0: Feather blend width
    Mapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes, 
           int blend=128, bool enable_gain_compensator=true, 
           cv::Size scale_output=cv::Size(0, 0));
    void stitch(std::vector<GpuMat> & inputs, GpuMat & output, 
                GpuMat & preview_output, bool mix_input_channels=true);
};

}

#endif
