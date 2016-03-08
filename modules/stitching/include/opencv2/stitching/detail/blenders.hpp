/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_STITCHING_BLENDERS_HPP__
#define __OPENCV_STITCHING_BLENDERS_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

namespace cv {
namespace detail {

//! @addtogroup stitching_blend
//! @{

/** @brief Base class for all blenders.

Simple blender which puts one image over another
*/
class CV_EXPORTS Blender
{
public:
    virtual ~Blender() {}

    enum { NO, FEATHER, MULTI_BAND };
    static Ptr<Blender> createDefault(int type, bool try_gpu = false);

    /** @brief Prepares the blender for blending.

    @param corners Source images top-left corners
    @param sizes Source image sizes
     */
    void prepare(const std::vector<Point> &corners, const std::vector<Size> &sizes);
    /** @overload */
    virtual void prepare(Rect dst_roi);
    /** @brief Processes the image.

    @param img Source image
    @param mask Source image mask
    @param tl Source image top-left corners
     */
    virtual void feed(InputArray img, InputArray mask, Point tl);
    /** @brief Blends and returns the final pano.

    @param dst Final pano
    @param dst_mask Final pano mask
     */
    virtual void blend(InputOutputArray dst, InputOutputArray dst_mask);

protected:
    UMat dst_, dst_mask_;
    Rect dst_roi_;
};

/** @brief Simple blender which mixes images at its borders.
 */
class CV_EXPORTS FeatherBlender : public Blender
{
public:
    FeatherBlender(float sharpness = 0.02f);

    float sharpness() const { return sharpness_; }
    void setSharpness(float val) { sharpness_ = val; }

    void prepare(Rect dst_roi);
    void feed(InputArray img, InputArray mask, Point tl);
    void blend(InputOutputArray dst, InputOutputArray dst_mask);

    //! Creates weight maps for fixed set of source images by their masks and top-left corners.
    //! Final image can be obtained by simple weighting of the source images.
    Rect createWeightMaps(const std::vector<UMat> &masks, const std::vector<Point> &corners,
                          std::vector<UMat> &weight_maps);

private:
    float sharpness_;
    UMat weight_map_;
    UMat dst_weight_map_;
};

inline FeatherBlender::FeatherBlender(float _sharpness) { setSharpness(_sharpness); }

/** @brief Blender which uses multi-band blending algorithm (see @cite BA83).
 */
class CV_EXPORTS MultiBandBlender : public Blender
{
public:
    MultiBandBlender(int try_gpu = false, int num_bands = 5, int weight_type = CV_32F);

    int numBands() const { return actual_num_bands_; }
    void setNumBands(int val) { actual_num_bands_ = val; }

    void prepare(Rect dst_roi);
    void feed(InputArray img, InputArray mask, Point tl);
    void blend(InputOutputArray dst, InputOutputArray dst_mask);

private:
    int actual_num_bands_, num_bands_;
    std::vector<UMat> dst_pyr_laplace_;
    std::vector<UMat> dst_band_weights_;
    Rect dst_roi_final_;
    bool can_use_gpu_;
    int weight_type_; //CV_32F or CV_16S
};


class CV_EXPORTS GPUStaticBlender {
protected:
    int num_images;
    std::vector<cv::Rect> rois;
    cv::Rect result_roi;
public:
    GPUStaticBlender(const std::vector<cuda::GpuMat> & masks, std::vector<cv::Rect> _rois);
    void blend(std::vector<cuda::GpuMat> &, cuda::GpuMat & dst);
    virtual ~GPUStaticBlender() {}
protected:
    virtual void do_blend(std::vector<cuda::GpuMat> &, cuda::GpuMat & dst) = 0;
};

class CV_EXPORTS MultiBandGPUBlender: public GPUStaticBlender {
private:
    int num_bands;
    std::vector<cuda::GpuMat> dst_pyr_laplace, dst_band_weights;

    std::vector<std::vector<cuda::GpuMat> > weight_pyr_gauss_lists;

    std::vector<cv::Rect> align_rois;
    cv::Rect align_result_roi;

private:
    std::vector<cuda::Stream> streams;
    std::vector<cuda::GpuMat> tmps;
    std::vector<cuda::GpuMat> dst_tmps;
    std::vector<std::vector<cuda::GpuMat> > src_pyr_laplaces;

public:
    MultiBandGPUBlender(const std::vector<cuda::GpuMat> & masks, 
                        std::vector<cv::Rect> _rois,
                        int num_bands_=5);
    void do_blend(std::vector<cuda::GpuMat> & imgs, cuda::GpuMat & dst) override;
};

class CV_EXPORTS FeatherGPUBlender: public GPUStaticBlender {
private:
    cuda::Stream stream;
    std::vector<cuda::GpuMat> weight_maps;
    cuda::GpuMat dst_16s;
public:
    FeatherGPUBlender(const std::vector<cuda::GpuMat> & masks, 
                      std::vector<cv::Rect> _rois, 
                      int border);
    void do_blend(std::vector<cuda::GpuMat> & imgs, cuda::GpuMat & dst) override;
};


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

void CV_EXPORTS normalizeUsingWeightMap(InputArray weight, InputOutputArray src);

void CV_EXPORTS createWeightMap(InputArray mask, float sharpness, InputOutputArray weight);

void CV_EXPORTS createLaplacePyr(InputArray img, int num_levels, std::vector<UMat>& pyr);
void CV_EXPORTS createLaplacePyrGpu(InputArray img, int num_levels, std::vector<UMat>& pyr);

// Restores source image
void CV_EXPORTS restoreImageFromLaplacePyr(std::vector<UMat>& pyr);
void CV_EXPORTS restoreImageFromLaplacePyrGpu(std::vector<UMat>& pyr);

//! @}

} // namespace detail
} // namespace cv

#endif // __OPENCV_STITCHING_BLENDERS_HPP__
