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

#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <thread>
#include <string>

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

    int sock = -1;
    bool ifSocket = true;
    std::string remote_addr = std::string("192.168.1.103");
    int remote_port = 23456;

private:
    int64_t first_time = 0;
    uint64_t getNowPts();

private:
    vr::Queue<cv::UMat *> full_q, empty_q;
    void feed(cv::UMat * frame);
    void run();

    std::thread th;

public:
    MonkeyEncoder(int width, int height, int bitrate, const char * filename, bool ifSocket, const char * remote_addr, int remote_port);
    ~MonkeyEncoder();

    void push(cv::UMat * frame);
    cv::UMat * pop();

    void start() {
        this->th = std::thread(&MonkeyEncoder::run, this);
    }
};

#endif
