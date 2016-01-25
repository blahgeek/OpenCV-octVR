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
}

void MonkeyVR::onStop(int index) {
    LOGD("onStop(%d)", index);
}

int MonkeyVR::onFrame(int index, cv::UMat * in, cv::Mat * out) {
    LOGD("onFrame(%d)", index);
    CV_Assert(index < 2);

    vr::Timer timer("onFrame");

    cv::cvtColor(*in, this->rgba_frame[index], cv::COLOR_YUV2RGBA_NV21, 4);
    timer.tick("cvtColor");
    this->rgba_frame[index].copyTo(*out);
    timer.tick("DtoH");
    // cv::cvtColor(*in, *out, cv::COLOR_YUV2RGBA_NV21, 4);
    // timer.tick("cvtColor(mat)");

    return 0;
}
