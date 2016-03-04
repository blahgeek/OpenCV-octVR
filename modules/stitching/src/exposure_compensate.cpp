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

#include "precomp.hpp"
#include <iostream>
#include "opencv2/core/private.cuda.hpp"

#if defined HAVE_CUDA && !defined(CUDA_DISABLER)
#include "opencv2/core/cuda_stream_accessor.hpp"

namespace cv { namespace cuda { namespace device {
    void mul_scalar_with_mask(const GpuMat &, float scale, const GpuMat &, GpuMat &, cudaStream_t stream);
}}}
#endif


namespace cv {
namespace detail {

Ptr<ExposureCompensator> ExposureCompensator::createDefault(int type)
{
    if (type == NO)
        return makePtr<NoExposureCompensator>();
    if (type == GAIN)
        return makePtr<GainCompensator>();
    if (type == GAIN_BLOCKS)
        return makePtr<BlocksGainCompensator>();
    CV_Error(Error::StsBadArg, "unsupported exposure compensation method");
    return Ptr<ExposureCompensator>();
}


void ExposureCompensator::feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
                               const std::vector<UMat> &masks)
{
    std::vector<std::pair<UMat,uchar> > level_masks;
    for (size_t i = 0; i < masks.size(); ++i)
        level_masks.push_back(std::make_pair(masks[i], 255));
    feed(corners, images, level_masks);
}


void GainCompensator::feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
                           const std::vector<std::pair<UMat,uchar> > &masks)
{
    LOGLN("Exposure compensation...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    const int num_images = static_cast<int>(images.size());
    Mat_<int> N(num_images, num_images); N.setTo(0);
    Mat_<double> I(num_images, num_images); I.setTo(0);

    //Rect dst_roi = resultRoi(corners, images);
    Mat subimg1, subimg2;
    Mat_<uchar> submask1, submask2, intersect;

    for (int i = 0; i < num_images; ++i)
    {
        for (int j = i; j < num_images; ++j)
        {
            Rect roi;
            if (overlapRoi(corners[i], corners[j], images[i].size(), images[j].size(), roi))
            {
                subimg1 = images[i](Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ);
                subimg2 = images[j](Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ);

                submask1 = masks[i].first(Rect(roi.tl() - corners[i], roi.br() - corners[i])).getMat(ACCESS_READ);
                submask2 = masks[j].first(Rect(roi.tl() - corners[j], roi.br() - corners[j])).getMat(ACCESS_READ);
                intersect = (submask1 == masks[i].second) & (submask2 == masks[j].second);

                N(i, j) = N(j, i) = std::max(1, countNonZero(intersect));

                double Isum1 = 0, Isum2 = 0;
                for (int y = 0; y < roi.height; ++y)
                {
                    const Point3_<uchar>* r1 = subimg1.ptr<Point3_<uchar> >(y);
                    const Point3_<uchar>* r2 = subimg2.ptr<Point3_<uchar> >(y);
                    for (int x = 0; x < roi.width; ++x)
                    {
                        if (intersect(y, x))
                        {
                            Isum1 += std::sqrt(static_cast<double>(sqr(r1[x].x) + sqr(r1[x].y) + sqr(r1[x].z)));
                            Isum2 += std::sqrt(static_cast<double>(sqr(r2[x].x) + sqr(r2[x].y) + sqr(r2[x].z)));
                        }
                    }
                }
                I(i, j) = Isum1 / N(i, j);
                I(j, i) = Isum2 / N(i, j);
            }
        }
    }

    double alpha = 0.01;
    double beta = 100;

    Mat_<double> A(num_images, num_images); A.setTo(0);
    Mat_<double> b(num_images, 1); b.setTo(0);
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            b(i, 0) += beta * N(i, j);
            A(i, i) += beta * N(i, j);
            if (j == i) continue;
            A(i, i) += 2 * alpha * I(i, j) * I(i, j) * N(i, j);
            A(i, j) -= 2 * alpha * I(i, j) * I(j, i) * N(i, j);
        }
    }

    solve(A, b, gains_);

    LOGLN("Exposure compensation, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)
GainCompensatorGPU::GainCompensatorGPU(const std::vector<cv::cuda::GpuMat> &) {
    throw_no_cuda();
}

void GainCompensatorGPU::feed(const std::vector<cv::cuda::GpuMat> &) {
    throw_no_cuda();
}

void GainCompensatorGPU::apply(std::vector<cv::cuda::GpuMat> &, std::vector<cv::cuda::GpuMat> &) {
    throw_no_cuda();
}

#else

GainCompensatorGPU::GainCompensatorGPU(const std::vector<cv::cuda::GpuMat> & masks) {
    this->num_images = masks.size();
    std::cerr << "Initing GainCompensatorGPU, num_images = " << num_images << std::endl;

    this->N = Mat_<int>(num_images, num_images); N.setTo(0);

    for(int i = 0 ; i < num_images ; i += 1) {
        for(int j = i ; j < num_images ; j += 1) {
            cv::cuda::GpuMat this_intersect;
            if(i != j) {
                cv::cuda::bitwise_and(masks[i], masks[j], this_intersect);
                int nz = cv::cuda::countNonZero(this_intersect);
                N(i, j) = N(j, i) = std::max(1, nz);
                this->intersects.push_back(this_intersect);
            } else {
                int nz = cv::cuda::countNonZero(masks[i]);
                N(i, j) = std::max(1, nz);
            }
        }
    }

    this->norm_images.resize(num_images);
    for(int i = 0 ; i < num_images ; i += 1)
        norm_images[i].create(masks[i].size(), CV_32F);

    this->intersect_count = (num_images * (num_images - 1)) / 2;
    this->streams.resize(std::max(intersect_count, num_images));
    for(int i = 0 ; i < intersect_count ; i += 1) {
        sum1_results_host.push_back(cv::cuda::HostMem(1, 1, CV_64FC1, cv::cuda::HostMem::SHARED));
        sum2_results_host.push_back(cv::cuda::HostMem(1, 1, CV_64FC1, cv::cuda::HostMem::SHARED));
        sum1_results.push_back(sum1_results_host.back().createGpuMatHeader());
        sum2_results.push_back(sum2_results_host.back().createGpuMatHeader());
        //sum1_results.push_back(cv::cuda::GpuMat(1, 1, CV_64FC1));
        //sum2_results.push_back(cv::cuda::GpuMat(1, 1, CV_64FC1));
    }
}

void GainCompensatorGPU::feed(const std::vector<cv::cuda::GpuMat> & images) {
    CV_Assert(images.size() == this->num_images);

    for(int i = 0 ; i < images.size() ; i += 1) {
        CV_Assert(images[i].type() == CV_8UC4 || images[i].type() == CV_8UC3);
        CV_Assert(images[i].size() == this->norm_images[i].size());
        images[i].elementNorm(norm_images[i], CV_32F, streams[i]);
    }
    for(int i = 0 ; i < num_images ; i += 1)
        streams[i].waitForCompletion();

    Mat_<double> I(num_images, num_images); I.setTo(0);

    int index = -1;
    for(int i = 0 ; i < images.size() ; i += 1) {
        for(int j = i + 1 ; j < images.size() ; j += 1) {
            index += 1;

            cv::cuda::GpuMat & s1 = sum1_results[index];
            cv::cuda::GpuMat & s2 = sum2_results[index];
            cv::cuda::calcSum(norm_images[i], s1, intersects[index], streams[index]);
            cv::cuda::calcSum(norm_images[j], s2, intersects[index], streams[index]);

            //s1.download(sum1_results_host[index], streams[index]);
            //s2.download(sum2_results_host[index], streams[index]);
        }
    }

    index = -1;
    for(int i = 0 ; i < images.size() ; i += 1) {
        for(int j = i + 1 ; j < images.size() ; j += 1) {
            index += 1;
            streams[index].waitForCompletion();

            int n = N(i, j);

            I(i, j) = *(double *)sum1_results_host[index].data / n;
            I(j, i) = *(double *)sum2_results_host[index].data / n;
        }
    }

    double alpha = 0.01;
    double beta = 100;

    Mat_<double> A(num_images, num_images); A.setTo(0);
    Mat_<double> b(num_images, 1); b.setTo(0);
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            b(i, 0) += beta * N(i, j);
            A(i, i) += beta * N(i, j);
            if (j == i) continue;
            A(i, i) += 2 * alpha * I(i, j) * I(i, j) * N(i, j);
            A(i, j) -= 2 * alpha * I(i, j) * I(j, i) * N(i, j);
        }
    }

    solve(A, b, gains_);
}

void GainCompensatorGPU::apply(std::vector<cv::cuda::GpuMat> & imgs, std::vector<cv::cuda::GpuMat> & masks) {
    CV_Assert(imgs.size() == num_images);
    CV_Assert(masks.size() == num_images);
    for(int i = 0 ; i < num_images ; i += 1)
        cv::cuda::device::mul_scalar_with_mask(imgs[i], gains_(i, 0), masks[i], imgs[i], 
                                               cuda::StreamAccessor::getStream(streams[i]));
    for(int i = 0 ; i < num_images ; i += 1)
        streams[i].waitForCompletion();
}

#endif

std::vector<double> GainCompensatorGPU::gains() const
{
    std::vector<double> gains_vec(gains_.rows);
    for (int i = 0; i < gains_.rows; ++i)
        gains_vec[i] = gains_(i, 0);
    return gains_vec;
}

void GainCompensator::apply(int index, Point /*corner*/, InputOutputArray image, InputArray /*mask*/)
{
    multiply(image, gains_(index, 0), image);
}


std::vector<double> GainCompensator::gains() const
{
    std::vector<double> gains_vec(gains_.rows);
    for (int i = 0; i < gains_.rows; ++i)
        gains_vec[i] = gains_(i, 0);
    return gains_vec;
}


void BlocksGainCompensator::feed(const std::vector<Point> &corners, const std::vector<UMat> &images,
                                     const std::vector<std::pair<UMat,uchar> > &masks)
{
    CV_Assert(corners.size() == images.size() && images.size() == masks.size());

    const int num_images = static_cast<int>(images.size());

    std::vector<Size> bl_per_imgs(num_images);
    std::vector<Point> block_corners;
    std::vector<UMat> block_images;
    std::vector<std::pair<UMat,uchar> > block_masks;

    // Construct blocks for gain compensator
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        Size bl_per_img((images[img_idx].cols + bl_width_ - 1) / bl_width_,
                        (images[img_idx].rows + bl_height_ - 1) / bl_height_);
        int bl_width = (images[img_idx].cols + bl_per_img.width - 1) / bl_per_img.width;
        int bl_height = (images[img_idx].rows + bl_per_img.height - 1) / bl_per_img.height;
        bl_per_imgs[img_idx] = bl_per_img;
        for (int by = 0; by < bl_per_img.height; ++by)
        {
            for (int bx = 0; bx < bl_per_img.width; ++bx)
            {
                Point bl_tl(bx * bl_width, by * bl_height);
                Point bl_br(std::min(bl_tl.x + bl_width, images[img_idx].cols),
                            std::min(bl_tl.y + bl_height, images[img_idx].rows));

                block_corners.push_back(corners[img_idx] + bl_tl);
                block_images.push_back(images[img_idx](Rect(bl_tl, bl_br)));
                block_masks.push_back(std::make_pair(masks[img_idx].first(Rect(bl_tl, bl_br)),
                                                masks[img_idx].second));
            }
        }
    }

    GainCompensator compensator;
    compensator.feed(block_corners, block_images, block_masks);
    std::vector<double> gains = compensator.gains();
    gain_maps_.resize(num_images);

    Mat_<float> ker(1, 3);
    ker(0,0) = 0.25; ker(0,1) = 0.5; ker(0,2) = 0.25;

    int bl_idx = 0;
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        Size bl_per_img = bl_per_imgs[img_idx];
        gain_maps_[img_idx].create(bl_per_img, CV_32F);

        {
            Mat_<float> gain_map = gain_maps_[img_idx].getMat(ACCESS_WRITE);
            for (int by = 0; by < bl_per_img.height; ++by)
                for (int bx = 0; bx < bl_per_img.width; ++bx, ++bl_idx)
                    gain_map(by, bx) = static_cast<float>(gains[bl_idx]);
        }

        sepFilter2D(gain_maps_[img_idx], gain_maps_[img_idx], CV_32F, ker, ker);
        sepFilter2D(gain_maps_[img_idx], gain_maps_[img_idx], CV_32F, ker, ker);
    }
}


void BlocksGainCompensator::apply(int index, Point /*corner*/, InputOutputArray _image, InputArray /*mask*/)
{
    CV_Assert(_image.type() == CV_8UC3);

    UMat u_gain_map;
    if (gain_maps_[index].size() == _image.size())
        u_gain_map = gain_maps_[index];
    else
        resize(gain_maps_[index], u_gain_map, _image.size(), 0, 0, INTER_LINEAR);

    Mat_<float> gain_map = u_gain_map.getMat(ACCESS_READ);
    Mat image = _image.getMat();
    for (int y = 0; y < image.rows; ++y)
    {
        const float* gain_row = gain_map.ptr<float>(y);
        Point3_<uchar>* row = image.ptr<Point3_<uchar> >(y);
        for (int x = 0; x < image.cols; ++x)
        {
            row[x].x = saturate_cast<uchar>(row[x].x * gain_row[x]);
            row[x].y = saturate_cast<uchar>(row[x].y * gain_row[x]);
            row[x].z = saturate_cast<uchar>(row[x].z * gain_row[x]);
        }
    }
}


} // namespace detail
} // namespace cv
