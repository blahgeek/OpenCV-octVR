/* 
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-25
*/

#ifndef ANDROID_JNI_MONKEY_H
#define ANDROID_JNI_MONKEY_H value

#include "./common.h"

#include <mutex>
#include <condition_variable>

#include <octvr.hpp>
#include "./codec.hpp"

class MonkeyVR {
private:
    static std::mutex _instance_mtx;
    static MonkeyVR * _instance;
public:
    static MonkeyVR * getInstance();

private:
    std::mutex mtx;
    std::condition_variable cond_empty, cond_full;
    volatile cv::UMat * waiting_frame = nullptr;
    cv::UMat rgba_frame[2];

    MonkeyEncoder * encoder = nullptr;

private:
    MonkeyVR();
public:
    void onStart(int index, int width, int height);
    void onStop(int index);
    int onFrame(int index, cv::UMat * in, cv::Mat * out);

private:
    int processTwoFrame(cv::UMat * front, cv::UMat *back, cv::Mat *out);
};

#endif
