/* 
* @Author: BlahGeek
* @Date:   2016-01-25
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-25
*/

#ifndef ANDROID_JNI_CODEC_H
#define ANDROID_JNI_CODEC_H value

#include "./common.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

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

public:
    MonkeyEncoder(int width, int height, int bitrate, const char * filename);
    ~MonkeyEncoder();
    void feed(cv::UMat * frame);
};

#endif
