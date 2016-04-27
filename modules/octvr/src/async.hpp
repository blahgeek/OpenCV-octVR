/* 
* @Author: BlahGeek
* @Date:   2015-12-01
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-26
*/

#ifndef LIBMAP_ASYNC_H_
#define LIBMAP_ASYNC_H_ value

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "octvr.hpp"
#include "./mapper.hpp"

#include <queue>
#include <mutex>
#include <thread>
#include <memory>
#include <condition_variable>

#ifdef HAVE_QT5
#include <QSharedMemory>
#endif

namespace vr {

class AsyncMultiMapperImpl: public AsyncMultiMapper {
private:
    std::vector<int> gain_modes;
    std::vector<cv::Rect_<double>> output_regions;
    int input_pix_fmt = OCTVR_UYVY422;
private:
    Queue<std::vector<cv::Mat>> inputs_mat_q;
    Queue<std::vector<cv::cuda::HostMem>> inputs_hostmem_q, free_inputs_hostmem_q;
    Queue<std::vector<cv::cuda::GpuMat>> inputs_gpumat_q, free_inputs_gpumat_q;

    Queue<std::vector<cv::cuda::GpuMat>> outputs_gpumat_q, free_outputs_gpumat_q;
    Queue<std::vector<cv::cuda::HostMem>> outputs_hostmem_q, free_outputs_hostmem_q;
    Queue<std::vector<cv::Mat>> outputs_mat_q, free_outputs_mat_q;

    // preview in RGB, using shared memory
    Queue<cv::cuda::GpuMat> previews_gpumat_q, free_previews_gpumat_q;
    Queue<cv::cuda::HostMem> previews_hostmem_q, free_previews_hostmem_q;
#ifdef HAVE_QT5
    QSharedMemory preview_data0, preview_data1, preview_meta;
#endif
    cv::Size preview_size;

    vr::Timer fps_timer;
    int64_t frame_count = 0;
    double frame_total_time = 0;
    double frame_fps = 0;

    cv::Size out_size; // merged
    std::vector<cv::Size> out_sizes;
    std::vector<cv::Size> in_sizes;

    std::vector<std::unique_ptr<Mapper>> mappers;

private:
    cv::cuda::Stream upload_stream, download_stream;
    std::vector<std::thread> running_threads;

private:
    void run_copy_inputs_mat_to_hostmem();
    void run_upload_inputs_hostmem_to_gpumat();
    void run_do_mapping();
    void run_download_outputs_gpumat_to_hostmem();
    void run_copy_outputs_hostmem_to_mat();

public:
    AsyncMultiMapperImpl(const std::vector<MapperTemplate> & mt,
                         std::vector<cv::Size> in_sizes, 
                         cv::Size out_size,
                         std::vector<int> blend_modes,
                         std::vector<int> gain_modes, // -1 means don't do it, N (self id) means do it, [0, N) means copy from No.x
                         std::vector<cv::Rect_<double>> output_regions,
                         int input_pix_fmt,
                         cv::Size preview_size);

    void push(std::vector<cv::Mat> & inputs, 
              cv::Mat & output) override;
    void pop() override;

    ~AsyncMultiMapperImpl() {
        mappers.clear();
        running_threads.clear();
    }
};

}

#endif
