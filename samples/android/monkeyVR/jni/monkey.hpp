/* 
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-22
*/

#ifndef ANDROID_JNI_MONKEY_H
#define ANDROID_JNI_MONKEY_H value

#define LOG_TAG "MonkeyVR"

#ifdef ANDROID
    #include <jni.h>
    #include <android/log.h>
    #define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
    #define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#else
    #define LOGD(...)
    #define LOGE(...)
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include <mutex>

#include <octvr.hpp>

class MonkeyVR {
private:
    static std::mutex _instance_mtx;
    static MonkeyVR * _instance;
public:
    static MonkeyVR * getInstance();

private:
    cv::UMat raw_frame[2];
    cv::UMat rgba_frame[2];

private:
    MonkeyVR();
public:
    void onStart(int index, int width, int height);
    void onStop(int index);
    int onFrame(int index, cv::Mat * in, cv::Mat * out);
};

#endif
