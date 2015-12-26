/* 
* @Author: BlahGeek
* @Date:   2015-12-01
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-07
*/

#ifndef LIBMAP_ASYNC_H_
#define LIBMAP_ASYNC_H_ value

#include "./libmap.hpp"
#include "./libmap_impl.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <queue>
#include <mutex>
#include <thread>
#include <memory>
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
    bool do_blend = true;
private:
    Queue<std::vector<cv::Mat>> inputs_mat_q;
    Queue<std::vector<cv::cuda::HostMem>> inputs_hostmem_q, free_inputs_hostmem_q;
    Queue<std::vector<cv::cuda::GpuMat>> inputs_gpumat_q, free_inputs_gpumat_q;

    Queue<std::vector<cv::cuda::GpuMat>> outputs_gpumat_q, free_outputs_gpumat_q;
    Queue<std::vector<cv::cuda::HostMem>> outputs_hostmem_q, free_outputs_hostmem_q;
    Queue<std::vector<cv::Mat>> outputs_mat_q, free_outputs_mat_q;

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
    AsyncMultiMapperImpl(const std::vector<MapperTemplate> & mt, std::vector<cv::Size> in_sizes, int blend=128);

    void push(std::vector<cv::Mat> & inputs, 
              std::vector<cv::Mat> & outputs) override;
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
