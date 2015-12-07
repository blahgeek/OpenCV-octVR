/* 
* @Author: BlahGeek
* @Date:   2015-12-01
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-07
*/


#include "./libmap_async.hpp"
#include <iostream>
#include <thread>

using namespace vr;

void AsyncMultiMapperImpl::run_copy_inputs_mat_to_hostmem() {
    auto inputs_mat = this->inputs_mat_q.pop();
    auto hostmems = this->free_inputs_hostmem_q.pop();

    for(int i = 0 ; i < inputs_mat.size() ; i += 1) {
        assert(inputs_mat[i].size() == hostmems[i].size());
        assert(inputs_mat[i].type() == hostmems[i].type());
        inputs_mat[i].copyTo(hostmems[i]);
    }
    this->inputs_hostmem_q.push(std::move(hostmems));
}

void AsyncMultiMapperImpl::run_upload_inputs_hostmem_to_gpumat() {
    auto hostmems = this->inputs_hostmem_q.pop();
    auto gpumats = this->free_inputs_gpumat_q.pop();

    for(int i = 0 ; i < hostmems.size() ; i += 1)
        gpumats[i].upload(hostmems[i], this->upload_stream);
    this->upload_stream.waitForCompletion();

    this->inputs_gpumat_q.push(std::move(gpumats));
    this->free_inputs_hostmem_q.push(std::move(hostmems));
}

void AsyncMultiMapperImpl::run_do_mapping() {
    auto gpumats = this->inputs_gpumat_q.pop();
    auto output = this->free_output_gpumat_q.pop();

    this->mapper->stitch(gpumats, output);

    this->output_gpumat_q.push(std::move(output));
    this->free_inputs_gpumat_q.push(std::move(gpumats));
}

void AsyncMultiMapperImpl::run_download_output_gpumat_to_hostmem() {
    auto gpumat = this->output_gpumat_q.pop();
    auto hostmem = this->free_output_hostmem_q.pop();

    // cv::cuda::HostMem hm;
    gpumat.download(hostmem, this->download_stream);
    this->download_stream.waitForCompletion();

    this->output_hostmem_q.push(std::move(hostmem));
    this->free_output_gpumat_q.push(std::move(gpumat));
}

void AsyncMultiMapperImpl::run_copy_output_hostmem_to_mat() {
    auto mat = this->free_output_mat_q.pop();
    auto hostmem = this->output_hostmem_q.pop();

    hostmem.createMatHeader().copyTo(mat);

    this->output_mat_q.push(std::move(mat));
    this->free_output_hostmem_q.push(std::move(hostmem));
}

void AsyncMultiMapperImpl::push(std::vector<cv::Mat> & inputs, cv::Mat & output) {
    this->inputs_mat_q.push(inputs);
    this->free_output_mat_q.push(output);
}

cv::Mat AsyncMultiMapperImpl::pop() {
    return this->output_mat_q.pop();
}

AsyncMultiMapper * AsyncMultiMapper::New(const MapperTemplate & m, std::vector<cv::Size> in_sizes) {
    return static_cast<AsyncMultiMapper *>(new AsyncMultiMapperImpl(m, in_sizes));
}

AsyncMultiMapperImpl::AsyncMultiMapperImpl(const MapperTemplate & mt, std::vector<cv::Size> in_sizes) {
    this->mapper = std::unique_ptr<Mapper>(new Mapper(mt, in_sizes));
    this->out_size = mt.out_size;
    this->in_sizes = in_sizes;

#define BUF_SIZE 3
    for(int n = 0 ; n < BUF_SIZE ; n += 1) {
        std::vector<cv::cuda::HostMem> inputs_hostmem;
        std::vector<cv::cuda::GpuMat> inputs_gpumat;
        for(auto & s: in_sizes) {
            inputs_hostmem.push_back(cv::cuda::HostMem(s, CV_8UC3));
            inputs_gpumat.push_back(cv::cuda::GpuMat(s, CV_8UC3));
        }
        free_inputs_hostmem_q.push(std::move(inputs_hostmem));
        free_inputs_gpumat_q.push(std::move(inputs_gpumat));
        free_output_gpumat_q.push(cv::cuda::GpuMat(out_size, CV_8UC3));
        free_output_hostmem_q.push(cv::cuda::HostMem(out_size, CV_8UC3));
    }

#define RUN_THREAD(X) do { \
    std::cerr << "Running " << #X << std::endl; \
    auto th = std::thread([](AsyncMultiMapperImpl *p) { \
        while(true) p->X(); \
    }, this); \
    this->running_threads.push_back(std::move(th)); \
} while(false)

    RUN_THREAD(run_copy_inputs_mat_to_hostmem);
    RUN_THREAD(run_upload_inputs_hostmem_to_gpumat);
    RUN_THREAD(run_do_mapping);
    RUN_THREAD(run_download_output_gpumat_to_hostmem);
    RUN_THREAD(run_copy_output_hostmem_to_mat);
}
