/* 
* @Author: BlahGeek
* @Date:   2016-01-25
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-26
*/

#include "./codec.hpp"
#include <assert.h>

uint64_t MonkeyEncoder::getNowPts() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t now = (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
    if(first_time == 0)
        first_time = now;
    return now - first_time;
}

#define MIME_TYPE "video/avc"
// #define MIME_TYPE "video/3gpp"
#define FRAMERATE 30
#define I_FRAME_INTERVAL 10

#define COLOR_FormatYUV420SemiPlanar 21
#define COLOR_FormatYUV420Planar 19

MonkeyEncoder::MonkeyEncoder(int width, int height, int bitrate, const char * filename): 
mWidth(width), mHeight(height) {
    AMediaFormat * format = AMediaFormat_new();
    AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, MIME_TYPE);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, width);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT, height);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, COLOR_FormatYUV420SemiPlanar);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_BIT_RATE, bitrate);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_FRAME_RATE, FRAMERATE);
    AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL);

    LOGD("MediaFormat init: %s", AMediaFormat_toString(format));

    this->codec = AMediaCodec_createEncoderByType(MIME_TYPE);
    AMediaCodec_configure(this->codec, format, NULL, NULL, AMEDIACODEC_CONFIGURE_FLAG_ENCODE);
    AMediaCodec_start(this->codec);

    LOGD("Opening %s for output...", filename);
    this->output = fopen(filename, "wb");
    assert(this->output);

    this->muxer = AMediaMuxer_new(fileno(this->output), AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4);
    LOGD("MonkeyEncoder init done.");
}

void MonkeyEncoder::feed(cv::UMat * frame) {
    vr::Timer timer("Encoder::feed");

    ssize_t inputBufIndex = AMediaCodec_dequeueInputBuffer(this->codec, -1);
    LOGD("dequeueInputBuffer: %d", inputBufIndex);
    timer.tick("dequeueInputBuffer");

    if(inputBufIndex < 0)
        return;

    size_t inputBufferSize = 0;
    uint8_t * inputBuffer = AMediaCodec_getInputBuffer(this->codec, inputBufIndex, &inputBufferSize);
    LOGD("getInputBuffer: size = %d", inputBufferSize);
    timer.tick("getInputBuffer");

    if(frame != nullptr) {
        cv::Mat inputBuffer_m(mHeight + mHeight / 2, mWidth, CV_8U, inputBuffer);
        frame->copyTo(inputBuffer_m);
        timer.tick("copyTo inputBuffer");
        AMediaCodec_queueInputBuffer(this->codec, inputBufIndex, 
                                     0, frame->total(), getNowPts(),
                                     0);
        LOGD("queueInputBuffer, frame->total = %d", frame->total());
    } else {
        AMediaCodec_queueInputBuffer(this->codec, inputBufIndex,
                                     0, 0, getNowPts(),
                                     AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
    }

    frame_count += 1;
    timer.tick("queueInputBuffer");

    while(true) {
        size_t encoderStatus = AMediaCodec_dequeueOutputBuffer(this->codec, &this->bufferinfo, 0);
        LOGD("dequeueOutputBuffer: %d, bufferinfo: (%d, %d, %d, %d)", 
             encoderStatus, bufferinfo.offset, bufferinfo.size, bufferinfo.presentationTimeUs, bufferinfo.flags);
        if(encoderStatus == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
            if(frame != nullptr) {
                LOGD("try again later");
                break;
            } else {
                LOGD("end of stream");
                if(bufferinfo.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                    LOGD("Really end, return");
                    break;
                }
                continue;
            }
        }
        if(encoderStatus == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
            // not expected for an encoder
            assert(false);
        }
        if(encoderStatus == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
            // should happen before receiving buffers, and should only happen once
            assert(!this->mMuxerStarted);
            AMediaFormat* newFormat = AMediaCodec_getOutputFormat(this->codec);
            LOGD("New format: %s", AMediaFormat_toString(newFormat));

            this->mTrackIndex = AMediaMuxer_addTrack(muxer, newFormat);
            AMediaMuxer_start(this->muxer);
            this->mMuxerStarted = true;
        }
        assert(encoderStatus >= 0);

        size_t outputBufferSize = 0;
        uint8_t * outputBuffer = AMediaCodec_getOutputBuffer(this->codec, encoderStatus, &outputBufferSize);
        LOGD("getOutputBuffer: size = %d", outputBufferSize);
        timer.tick("getOutputBuffer");

        AMediaMuxer_writeSampleData(this->muxer, mTrackIndex, outputBuffer, &bufferinfo);
        timer.tick("writeSampleData");

        AMediaCodec_releaseOutputBuffer(this->codec, encoderStatus, false);
    }
}

MonkeyEncoder::~MonkeyEncoder() {
    if(codec) {
        LOGD("closing encoder");
        AMediaCodec_stop(codec);
        AMediaCodec_delete(codec);
    }
    if(muxer) {
        LOGD("closing muxer");
        AMediaMuxer_stop(muxer);
        AMediaMuxer_delete(muxer);
    }
    LOGD("closing output file");
    fclose(output);
}
