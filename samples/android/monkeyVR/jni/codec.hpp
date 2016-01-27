/* 
* @Author: BlahGeek
* @Date:   2016-01-25
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-27
*/

#ifndef ANDROID_JNI_CODEC_H
#define ANDROID_JNI_CODEC_H value

#include "./common.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include <thread>

#include <octvr.hpp>

class MonkeyEncoder {
private:
    AMediaCodec *codec = nullptr;
    AMediaMuxer *muxer = nullptr;
    AMediaCodecBufferInfo bufferinfo;

    bool mMuxerStarted = false;
    int mTrackIndex = -1;

    int frame_count = 0;
    int mWidth, mHeight;
    FILE * output = nullptr;

private:
    uint64_t first_time = 0;
    uint64_t getNowPts();

private:
    vr::Queue<cv::UMat *> full_q, empty_q;
    void feed(cv::UMat * frame);
    void run();

    std::thread th;

public:
    MonkeyEncoder(int width, int height, int bitrate, const char * filename);
    ~MonkeyEncoder();

    void push(cv::UMat * frame);
    cv::UMat * pop();

    void start() {
        this->th = std::thread(&MonkeyEncoder::run, this);
    }
};

#endif
