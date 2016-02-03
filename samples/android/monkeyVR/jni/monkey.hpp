/*
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-03
*/

#ifndef ANDROID_JNI_MONKEY_H
#define ANDROID_JNI_MONKEY_H value

#include "./common.h"

#include <string>
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

    cv::Size in_sizes[2];
    bool stopping = false;

    cv::UMat result[2];
    int encoding_result_index = -1;

    MonkeyEncoder * encoder = nullptr;
    vr::FastMapper * mapper = nullptr;

    int bitrate = 10000000;
    std::string outfile_path = std::string("/sdcard/octvr.mp4");
    std::string remote_addr = std::string("192.168.1.103");
    int remote_port = 23456;
    bool ifStitch = true;
    bool ifSocket = true;

    uint64_t frameCount = 0;

private:
    MonkeyVR();
public:
    void onStart(int index, int width, int height);
    void onStop(int index);
    int onFrame(int index, cv::UMat * in, cv::Mat * out);
    void setParams(int bitrate, const char * outfile_path,
                   const char * remote_addr, int remote_port,
                   bool ifStitch, bool ifSocket);
    std::string printParams();

private:
    int processTwoFrame(cv::UMat * back, cv::UMat *front, cv::Mat *out);
};

#endif
