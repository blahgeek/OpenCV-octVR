/*
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-03
*/

#include "./monkey.hpp"

#define INPUT_FILENAME "/sdcard/map.dat"

MonkeyVR * MonkeyVR::_instance = nullptr;
std::mutex MonkeyVR::_instance_mtx;

MonkeyVR * MonkeyVR::getInstance() {
    std::lock_guard<std::mutex> lock(MonkeyVR::_instance_mtx);
    if (!MonkeyVR::_instance)
        MonkeyVR::_instance = new MonkeyVR();
    return MonkeyVR::_instance;
}


MonkeyVR::MonkeyVR() {
    CV_Assert(cv::ocl::haveOpenCL());
    LOGD("MonkeyVR::MonkeyVR()");
}

void MonkeyVR::onStart(int index, int width, int height) {
    std::lock_guard<std::mutex> lock(mtx);

    CV_Assert(index == 0 || index == 1);
    LOGD("onStart(%d, %d, %d)", index, width, height);
    this->in_sizes[index] = cv::Size(width, height);

    if(this->in_sizes[0].area() > 0 && this->in_sizes[1].area() > 0) {
        LOGD("Both camera started, initing...");
        if (!this->ifStitch) {
            CV_Assert(in_sizes[0].width = in_sizes[1].width);
            encoder = new MonkeyEncoder(in_sizes[0].width,
                                        in_sizes[0].height + in_sizes[1].height,
                                        this->bitrate, this->outfile_path.c_str(),
                                        this->ifSocket, this->remote_addr.c_str(),
                                        this->remote_port);
        }
        else {
            LOGD("Loading map file %s", INPUT_FILENAME);
            std::ifstream map_file(INPUT_FILENAME);
            vr::MapperTemplate map_template(map_file);
            mapper = new vr::FastMapper(map_template, std::vector<cv::Size>(in_sizes, in_sizes+2));
            encoder = new MonkeyEncoder(map_template.out_size.width,
                                        map_template.out_size.height,
                                        this->bitrate, this->outfile_path.c_str(),
                                        this->ifSocket, this->remote_addr.c_str(),
                                        this->remote_port);
        }

        this->encoding_result_index = -1;
        this->stopping = false;
        encoder->start();
    }
}

void MonkeyVR::onStop(int index) {
    std::lock_guard<std::mutex> lock(mtx);

    CV_Assert(index == 0 || index == 1);
    LOGD("onStop(%d)", index);
    this->in_sizes[index] = cv::Size();

    this->stopping = true;
    this->cond_full.notify_all();
    this->cond_empty.notify_all();

    if(this->in_sizes[0].area() == 0 && this->in_sizes[0].area() == 0) {
        LOGD("Both camera stopped, cleaning...");
        if(encoder) {
            encoder->pop();
            encoder->push(nullptr);
            encoder->pop();
            delete encoder;
            encoder = nullptr;
        }
        if(mapper) {
            delete mapper;
            mapper = nullptr;
        }
    }
}

int MonkeyVR::onFrame(int index, cv::UMat * in, cv::Mat * out) {
    LOGD("onFrame(%d)", index);
    CV_Assert(index < 2);

    std::unique_lock<std::mutex> lock(mtx);
    if(index == 0) { // back
        this->waiting_frame = in;
        cond_full.notify_all();
        cond_empty.wait(lock, [this](){
            return this->waiting_frame == nullptr || this->stopping;
        });
        return 0;
    } else { // front
        cond_full.wait(lock, [this](){
            return this->waiting_frame != nullptr || this->stopping;
        });
        if(this->stopping)
            return 0;
        vr::Timer timer("onFrame");
        int ret = this->processTwoFrame((cv::UMat *)waiting_frame, in, out);
        timer.tick("processTwoFrame");

        this->waiting_frame = nullptr;
        cond_empty.notify_all();
        return ret;
    }

    return 0;
}

void MonkeyVR::setParams(int _bitrate, const char * _outfile_path,
               const char * _remote_addr, int _remote_port,
               bool _ifStitch, bool _ifSocket) {
    this->bitrate = _bitrate;
    this->outfile_path = std::string(_outfile_path);
    this->remote_addr = std::string(_remote_addr);
    this->remote_port = _remote_port;
    this->ifStitch = _ifStitch;
    this->ifSocket = _ifSocket;
}

std::string MonkeyVR::printParams() {
    char intOut[200];
    sprintf(intOut, "\nBitrate: %d\nRemote port: %d\nSocket: %c",
            this->bitrate, this->remote_port, (this->ifSocket ? 'Y' : 'N'));
    return std::string("Output path: " ) + this->outfile_path +
           std::string("\nRemote address: ") + this->remote_addr +
           std::string(intOut);
}

int MonkeyVR::processTwoFrame(cv::UMat * back, cv::UMat * front, cv::Mat * out) {
    vr::Timer timer("processTwoFrame");

    int stitch_target_index = 0;
    if(encoding_result_index >= 0)
        stitch_target_index = 1 - encoding_result_index;

    if (!this->ifStitch) {
        result[stitch_target_index].create(front->rows + back->rows, front->cols, CV_8U);
        cv::UMat ref = result[stitch_target_index].rowRange(0, in_sizes[0].height);
        back->rowRange(0, in_sizes[0].height).copyTo(ref);
        ref = result[stitch_target_index].rowRange(in_sizes[0].height, in_sizes[0].height + in_sizes[1].height);
        front->rowRange(0, in_sizes[1].height).copyTo(ref);
        ref = result[stitch_target_index].rowRange(in_sizes[0].height + in_sizes[1].height,
                                                   in_sizes[0].height + in_sizes[1].height + in_sizes[0].height / 2);
        back->rowRange(in_sizes[0].height, in_sizes[0].height + in_sizes[0].height / 2)
                     .copyTo(ref);
        ref = result[stitch_target_index].rowRange(in_sizes[0].height + in_sizes[1].height + in_sizes[0].height / 2,
                                                   in_sizes[0].height + in_sizes[1].height + in_sizes[0].height / 2 + in_sizes[1].height / 2);
        front->rowRange(in_sizes[1].height, in_sizes[1].height + in_sizes[1].height / 2)
                     .copyTo(ref);
    } else {
        mapper->stitch_nv12(std::vector<cv::UMat>({*back, *front}), result[stitch_target_index]);
    }

    // cv::ocl::finish();
    timer.tick("stitch_nv12");

    encoder->push(result + stitch_target_index);
    if(encoding_result_index >= 0) {
        cv::UMat * pop_ret = encoder->pop();
        CV_Assert(pop_ret == &result[encoding_result_index]);
    }
    encoding_result_index = stitch_target_index;
    timer.tick("encoder feed");

    if((frameCount++) % 5 == 0) {
        cv::UMat rgba, resized_rgba;
        cv::cvtColor(result[stitch_target_index], rgba, cv::COLOR_YUV2RGBA_NV21, 4);
        cv::resize(rgba, resized_rgba, this->in_sizes[1]);
        resized_rgba.copyTo(*out);
        return 1;
    }

    return 0;
}