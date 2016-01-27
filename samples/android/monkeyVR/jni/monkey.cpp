/* 
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-27
*/

#include "./monkey.hpp"

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
    LOGD("onStart(%d, %d, %d)", index, width, height);
    if(index == 0) {
        LOGD("Loading map file");
        std::ifstream map_file("/sdcard/map.dat");
        vr::MapperTemplate map_template(map_file);
        mapper = new vr::FastMapper(map_template, std::vector<cv::Size>(2, cv::Size(width, height))); // FIXME
        encoder = new MonkeyEncoder(map_template.out_size.width, 
                                    map_template.out_size.height, 
                                    5000000, "/sdcard/octvr.mp4");
        encoder->start();
    }
}

void MonkeyVR::onStop(int index) {
    LOGD("onStop(%d)", index);
    if(index == 1) {
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
    if(index == 0) {
        this->waiting_frame = in;
        cond_full.notify_all();
        cond_empty.wait(lock, [this](){
            return this->waiting_frame == nullptr;
        });
        return 0;
    } else {
        cond_full.wait(lock, [this](){
            return this->waiting_frame != nullptr;
        });
        vr::Timer timer("onFrame");
        int ret = this->processTwoFrame(in, (cv::UMat *)waiting_frame, out);
        timer.tick("processTwoFrame");

        this->waiting_frame = nullptr;
        cond_empty.notify_all();
        return ret;
    }

    return 0;
}

int MonkeyVR::processTwoFrame(cv::UMat * front, cv::UMat * back, cv::Mat * out) {
    vr::Timer timer("processTwoFrame");

    int stitch_target_index = 0;
    if(encoding_result_index >= 0)
        stitch_target_index = 1 - encoding_result_index;

    mapper->stitch_nv12(std::vector<cv::UMat>({*front, *back}), result[stitch_target_index]);
    cv::ocl::finish();
    timer.tick("stitch_nv12");

    encoder->push(result + stitch_target_index);
    if(encoding_result_index >= 0) {
        cv::UMat * pop_ret = encoder->pop();
        CV_Assert(pop_ret == &result[encoding_result_index]);
    }
    encoding_result_index = stitch_target_index;
    timer.tick("encoder feed");

    return 0;
}
