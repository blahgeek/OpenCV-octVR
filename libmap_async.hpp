/* 
* @Author: BlahGeek
* @Date:   2015-12-01
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-01
*/

#ifndef LIBMAP_ASYNC_H_
#define LIBMAP_ASYNC_H_ value

#include "./libmap.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace vr {

template <class T>
class Queue {
private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cond_empty;
public:
    bool empty() { return q.empty() ;}
    void push(T&& val) {
        std::lock_guard<std::mutex> guard(mtx);
        q.push(std::forward<T>(val));
        cond_empty.notify_one();
    }
    void push(const T & val) { this->push(T(val)); }
    T pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cond_empty.wait(lock, [this](){ return !this->empty(); });
        T ret = q.front();
        q.pop();
        return ret;
    }
};

class AsyncMultiMapperImpl: public AsyncMultiMapper {
private:
    Queue<std::vector<cv::Mat>> inputs_mat_q;
    Queue<std::vector<cv::cuda::HostMem>> inputs_hostmem_q;
    Queue<std::vector<cv::cuda::GpuMat>> inputs_gpumat_q;
    Queue<cv::cuda::GpuMat> output_gpumat_q;
    Queue<cv::cuda::HostMem> output_hostmem_q;
    Queue<cv::Mat> output_empty_mat_q;
    Queue<cv::Mat> output_mat_q;

    MultiMapper * mapper;

private:
    cv::cuda::Stream upload_stream, download_stream;

private:
    void run_copy_inputs_mat_to_hostmem();
    void run_upload_inputs_hostmem_to_gpumat();
    void run_do_mapping();
    void run_download_output_gpumat_to_hostmem();
    void run_copy_output_hostmem_to_mat();

public:
    AsyncMultiMapperImpl(MultiMapper * m);

    void push(std::vector<cv::Mat> & inputs, cv::Mat & output) override;
    cv::Mat pop() override;
};

}

#endif
