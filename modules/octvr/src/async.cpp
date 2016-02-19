/* 
* @Author: BlahGeek
* @Date:   2015-12-01
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-19
*/


#include "./async.hpp"
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
    auto outputs = this->free_outputs_gpumat_q.pop();

    for(int i = 0 ; i < outputs.size() ; i += 1)
        this->mappers[i]->stitch(gpumats, outputs[i]);

    this->outputs_gpumat_q.push(std::move(outputs));
    this->free_inputs_gpumat_q.push(std::move(gpumats));
}

void AsyncMultiMapperImpl::run_download_outputs_gpumat_to_hostmem() {
    auto gpumats = this->outputs_gpumat_q.pop();
    auto hostmems = this->free_outputs_hostmem_q.pop();

    for(int i = 0 ; i < gpumats.size() ; i += 1)
        gpumats[i].download(hostmems[i], this->download_stream);

    this->download_stream.waitForCompletion();

    this->outputs_hostmem_q.push(std::move(hostmems));
    this->free_outputs_gpumat_q.push(std::move(gpumats));
}

void AsyncMultiMapperImpl::run_copy_outputs_hostmem_to_mat() {
    auto mats = this->free_outputs_mat_q.pop();
    auto hostmems = this->outputs_hostmem_q.pop();

    for(int i = 0 ; i < mats.size() ; i += 1)
        hostmems[i].createMatHeader().copyTo(mats[i]);

    this->outputs_mat_q.push(std::move(mats));
    this->free_outputs_hostmem_q.push(std::move(hostmems));
}

void AsyncMultiMapperImpl::push(std::vector<cv::Mat> & inputs, 
                                std::vector<cv::Mat> & outputs) {
    assert(inputs.size() == in_sizes.size());
    assert(outputs.size() == out_sizes.size());

    this->inputs_mat_q.push(inputs);
    this->free_outputs_mat_q.push(outputs);
}

void AsyncMultiMapperImpl::push(std::vector<cv::Mat> & inputs, cv::Mat & output) {
    std::vector<cv::Mat> out({output});
    this->push(inputs, out);
}

void AsyncMultiMapperImpl::pop() {
    this->outputs_mat_q.pop();
}

AsyncMultiMapper * AsyncMultiMapper::New(const MapperTemplate & m, std::vector<cv::Size> in_sizes, int blend) {
    return AsyncMultiMapper::New(std::vector<MapperTemplate>({m}), in_sizes, blend);
}
AsyncMultiMapper * AsyncMultiMapper::New(const std::vector<MapperTemplate> & m, std::vector<cv::Size> in_sizes, int blend) {
    return static_cast<AsyncMultiMapper *>(new AsyncMultiMapperImpl(m, in_sizes, blend));
}

AsyncMultiMapperImpl::AsyncMultiMapperImpl(const std::vector<MapperTemplate> & mts, std::vector<cv::Size> in_sizes, int blend) {
    this->in_sizes = in_sizes;
    this->do_blend = (blend != 0);
    for(int i = 0 ; i < mts.size() ; i += 1) {
        this->mappers.emplace_back(new Mapper(mts[i], in_sizes, blend));
        this->out_sizes.push_back(mts[i].out_size);
    }

#define BUF_SIZE 3
    for(int n = 0 ; n < BUF_SIZE ; n += 1) {
        std::vector<cv::cuda::HostMem> inputs_hostmem;
        std::vector<cv::cuda::GpuMat> inputs_gpumat;
        for(auto & s: in_sizes) {
            inputs_hostmem.push_back(cv::cuda::HostMem(s, CV_8UC2));
            inputs_gpumat.push_back(cv::cuda::GpuMat(s, CV_8UC2));
        }
        free_inputs_hostmem_q.push(std::move(inputs_hostmem));
        free_inputs_gpumat_q.push(std::move(inputs_gpumat));

        std::vector<cv::cuda::GpuMat> outputs_gpumat;
        std::vector<cv::cuda::HostMem> outputs_hostmem;
        for(int i = 0 ; i < mts.size() ; i += 1) {
            outputs_gpumat.push_back(cv::cuda::GpuMat(out_sizes[i], CV_8UC2));
            outputs_hostmem.push_back(cv::cuda::HostMem(out_sizes[i], CV_8UC2));
        }
        free_outputs_gpumat_q.push(std::move(outputs_gpumat));
        free_outputs_hostmem_q.push(std::move(outputs_hostmem));
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
    RUN_THREAD(run_download_outputs_gpumat_to_hostmem);
    RUN_THREAD(run_copy_outputs_hostmem_to_mat);
}
