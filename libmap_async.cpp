/* 
* @Author: BlahGeek
* @Date:   2015-12-01
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-01
*/


#include "./libmap_async.hpp"
#include <iostream>
#include <thread>

using namespace vr;

void AsyncMultiMapperImpl::run_copy_inputs_mat_to_hostmem() {
    auto inputs_mat = this->inputs_mat_q.pop();
    std::vector<cv::cuda::HostMem> hostmems;
    for(auto & m: inputs_mat) {
        cv::cuda::HostMem hm;
        m.copyTo(hm);
        hostmems.push_back(std::move(hm));
    }
    this->inputs_hostmem_q.push(std::move(hostmems));
}

void AsyncMultiMapperImpl::run_upload_inputs_hostmem_to_gpumat() {
    auto hostmems = this->inputs_hostmem_q.pop();
    std::vector<cv::cuda::GpuMat> gpumats(hostmems.size());
    for(int i = 0 ; i < hostmems.size() ; i += 1)
        gpumats[i].upload(hostmems[i], this->upload_stream);
    this->upload_stream.waitForCompletion();
    this->inputs_gpumat_q.push(std::move(gpumats));
}

void AsyncMultiMapperImpl::run_do_mapping() {
    auto gpumats = this->inputs_gpumat_q.pop();
    cv::cuda::GpuMat output(mapper->get_output_size(), CV_8UC3);
    this->mapper->get_output(gpumats, output);
    this->output_gpumat_q.push(std::move(output));
}

void AsyncMultiMapperImpl::run_download_output_gpumat_to_hostmem() {
    auto gpumat = this->output_gpumat_q.pop();
    cv::cuda::HostMem hm;
    gpumat.download(hm, this->download_stream);
    this->download_stream.waitForCompletion();
    this->output_hostmem_q.push(std::move(hm));
}

void AsyncMultiMapperImpl::run_copy_output_hostmem_to_mat() {
    auto out_mat = this->output_empty_mat_q.pop();
    auto out_hostmem = this->output_hostmem_q.pop();
    out_hostmem.createMatHeader().copyTo(out_mat);
    this->output_mat_q.push(std::move(out_mat));
}

void AsyncMultiMapperImpl::push(std::vector<cv::Mat> & inputs, cv::Mat & output) {
    this->inputs_mat_q.push(inputs);
    this->output_empty_mat_q.push(output);
}

cv::Mat AsyncMultiMapperImpl::pop() {
    return this->output_mat_q.pop();
}

AsyncMultiMapper * AsyncMultiMapper::New(MultiMapper * m) {
    return static_cast<AsyncMultiMapper *>(new AsyncMultiMapperImpl(m));
}

AsyncMultiMapperImpl::AsyncMultiMapperImpl(MultiMapper * m) {
    this->mapper = m;

#define RUN_THREAD(X) do { \
    std::cerr << "Running " << #X << std::endl; \
    auto th = std::thread([](AsyncMultiMapperImpl *p) { \
        while(true) p->X(); \
    }, this); \
    th.detach(); \
} while(false)

    RUN_THREAD(run_copy_inputs_mat_to_hostmem);
    RUN_THREAD(run_upload_inputs_hostmem_to_gpumat);
    RUN_THREAD(run_do_mapping);
    RUN_THREAD(run_download_output_gpumat_to_hostmem);
    RUN_THREAD(run_copy_output_hostmem_to_mat);
}
