/* 
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-25
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
    this->rgba_frame[0] = cv::UMat(height, width, CV_8UC4, cv::USAGE_ALLOCATE_SHARED_MEMORY);
    this->rgba_frame[1] = cv::UMat(height, width, CV_8UC4, cv::USAGE_ALLOCATE_SHARED_MEMORY);
    if(index == 0)
        encoder = new MonkeyEncoder(width, height, 5000000, "/sdcard/octvr.mp4");
}

void MonkeyVR::onStop(int index) {
    LOGD("onStop(%d)", index);
    if(index == 0 && encoder) {
        encoder->feed(nullptr);
        delete encoder;
        encoder = nullptr;
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

    cv::UMat added;
    cv::add(*front, *back, added);
    timer.tick("add");

    encoder->feed(&added);
    timer.tick("encoder feed");

    return 0;
}
