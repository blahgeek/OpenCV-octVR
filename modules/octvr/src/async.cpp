/* 
* @Author: BlahGeek
* @Date:   2015-12-01
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-26
*/


#include "./async.hpp"
#include <iostream>
#include <thread>

using namespace vr;

static cv::Rect _rect_mul_size(cv::Rect_<double> region, cv::Size size) {
    int x = region.x * size.width;
    int y = region.y * size.height;
    int w = region.width * size.width;
    int h = region.height * size.height;
    if(x + w >= size.width)
        w = size.width - x;
    if(y + h >= size.height)
        h = size.height - y;
    return cv::Rect(x, y, w, h);
}

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
    auto preview_output = this->free_previews_gpumat_q.pop();

    std::vector<std::vector<double>> gains_list;

    // split preview output for every output
    for(int i = 0 ; i < outputs.size() ; i += 1) {
        auto preview = preview_output(_rect_mul_size(output_regions[i], preview_output.size()));
        std::vector<double> predefined_gain_list;
        if(gain_modes[i] < i && gain_modes[i] >= 0)
            predefined_gain_list = gains_list[gain_modes[i]];
        this->mappers[i]->stitch(gpumats, outputs[i], preview, 
                                 i == 0 && input_pix_fmt == OCTVR_UYVY422,
                                 predefined_gain_list);
        gains_list.push_back(this->mappers[i]->gains());
    }

    this->outputs_gpumat_q.push(std::move(outputs));
    this->free_inputs_gpumat_q.push(std::move(gpumats));
    this->previews_gpumat_q.push(std::move(preview_output));
}

void AsyncMultiMapperImpl::run_download_outputs_gpumat_to_hostmem() {
    auto gpumats = this->outputs_gpumat_q.pop();
    auto preview_gpumat = this->previews_gpumat_q.pop();
    auto hostmems = this->free_outputs_hostmem_q.pop();
    auto preview_hostmem = this->free_previews_hostmem_q.pop();

    for(int i = 0 ; i < gpumats.size() ; i += 1)
        gpumats[i].download(hostmems[i], this->download_stream);

    if(preview_size.area() > 0)
        preview_gpumat.download(preview_hostmem, this->download_stream);

    this->download_stream.waitForCompletion();

    this->outputs_hostmem_q.push(std::move(hostmems));
    this->free_outputs_gpumat_q.push(std::move(gpumats));
    this->previews_hostmem_q.push(std::move(preview_hostmem));
    this->free_previews_gpumat_q.push(std::move(preview_gpumat));
}

void AsyncMultiMapperImpl::run_copy_outputs_hostmem_to_mat() {
    auto mats = this->free_outputs_mat_q.pop();
    auto hostmems = this->outputs_hostmem_q.pop();

    for(int i = 0 ; i < mats.size() ; i += 1)
        hostmems[i].createMatHeader().copyTo(mats[i]);

    this->outputs_mat_q.push(std::move(mats));
    this->free_outputs_hostmem_q.push(std::move(hostmems));

    frame_total_time += fps_timer.tick("One frame poped");
    frame_count += 1;
    if(frame_count == 10) {
        frame_fps = 1.0 / (frame_total_time / frame_count / 1000);
        frame_count = 0;
        frame_total_time = 0;
    }

    auto preview_hostmem = this->previews_hostmem_q.pop();
    #ifdef HAVE_QT5
    if(preview_size.area() > 0) {
        void * preview_meta_p = preview_meta.data();
        char zone_index = *(static_cast<char *>(preview_meta_p));
        std::cerr << "Copying preview frame to shared zone " << int(zone_index);

        QSharedMemory & target = zone_index == 0 ? preview_data0 : preview_data1;
        target.lock();

        preview_hostmem.createMatHeader().copyTo(cv::Mat(preview_size, CV_8UC3,
                                                         static_cast<char *>(target.data()) + sizeof(struct PreviewDataHeader),
                                                         0));
        struct PreviewDataHeader * hdr = static_cast<struct PreviewDataHeader *>(target.data());
        hdr->width = preview_size.width;
        hdr->height = preview_size.height;
        hdr->step = 0;
        hdr->fps = this->frame_fps;

        target.unlock();
    }
    #endif
    this->free_previews_hostmem_q.push(std::move(preview_hostmem));
}

void AsyncMultiMapperImpl::push(std::vector<cv::Mat> & inputs, 
                                cv::Mat & output) {
    CV_Assert(inputs.size() == in_sizes.size());
    CV_Assert(output.size() == out_size);

    std::vector<cv::Mat> outputs;

    for(size_t i = 0 ; i < this->output_regions.size() ; i += 1)
        outputs.push_back(output(_rect_mul_size(output_regions[i], output.size())));

    this->inputs_mat_q.push(inputs);
    this->free_outputs_mat_q.push(outputs);
}

void AsyncMultiMapperImpl::pop() {
    this->outputs_mat_q.pop();
}

AsyncMultiMapper * AsyncMultiMapper::New(const std::vector<MapperTemplate> & m,
                                         std::vector<cv::Size> in_sizes, 
                                         cv::Size out_size,
                                         std::vector<int> blend_modes,
                                         std::vector<int> gain_modes,
                                         std::vector<cv::Rect_<double>> output_regions,
                                         int input_pix_fmt,
                                         cv::Size preview_size) {
    return static_cast<AsyncMultiMapper *>(new AsyncMultiMapperImpl(m,
                                                                    in_sizes,
                                                                    out_size,
                                                                    blend_modes,
                                                                    gain_modes,
                                                                    output_regions,
                                                                    input_pix_fmt,
                                                                    preview_size));
}

AsyncMultiMapperImpl::AsyncMultiMapperImpl(const std::vector<MapperTemplate> & mts,
                                           std::vector<cv::Size> in_sizes, 
                                           cv::Size out_size,
                                           std::vector<int> blend_modes,
                                           std::vector<int> gain_modes,
                                           std::vector<cv::Rect_<double>> output_regions,
                                           int input_pix_fmt,
                                           cv::Size _preview_size):
fps_timer("FPS Timer"){

    this->preview_size = cv::Size(0, 0);
#ifdef HAVE_QT5
    this->preview_size = _preview_size;
#endif

    this->input_pix_fmt = input_pix_fmt;
    this->in_sizes = in_sizes;
    this->out_size = out_size;
    this->gain_modes = gain_modes;
    this->output_regions = output_regions;

    for(int i = 0 ; i < mts.size() ; i += 1) {
        auto r = _rect_mul_size(output_regions[i], out_size);
        this->mappers.emplace_back(new Mapper(mts[i],
                                              in_sizes,
                                              blend_modes[i],
                                              gain_modes[i] >= 0,
                                              r.size()
                                              ));
        this->out_sizes.push_back(r.size());
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

        if(this->preview_size.area() > 0) {
            auto preview_mat = cv::cuda::GpuMat(preview_size, CV_8UC3);
            auto preview_mem = cv::cuda::HostMem(preview_size, CV_8UC3);
            free_previews_gpumat_q.push(std::move(preview_mat));
            free_previews_hostmem_q.push(std::move(preview_mem));
        } else {
            free_previews_gpumat_q.push(cv::cuda::GpuMat());
            free_previews_hostmem_q.push(cv::cuda::HostMem());
        }
    }

#ifdef HAVE_QT5
    if(this->preview_size.area() > 0) {

        std::cerr << "Preview size: " << this->preview_size << std::endl;

    #define ATTACH(x, key, s) \
        do { \
            (x).setKey(key); \
            bool ret = (x).attach(); \
            if(ret) break; \
            std::cerr << "Attach failed: " << (x).errorString().toStdString() << std::endl; \
            CV_Assert(ret); \
            CV_Assert((x).size() == s); \
        } while(0)

        ATTACH(preview_data0, OCTVR_PREVIEW_DATA0_MEMORY_KEY,
               sizeof(struct PreviewDataHeader) + preview_size.area());
        ATTACH(preview_data1, OCTVR_PREVIEW_DATA1_MEMORY_KEY,
               sizeof(struct PreviewDataHeader) + preview_size.area());
        ATTACH(preview_meta, OCTVR_PREVIEW_DATA_META_MEMORY_KEY, 1);

    #undef ATTACH
    }
#endif

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
