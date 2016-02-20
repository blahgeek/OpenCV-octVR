/* 
* @Author: BlahGeek
* @Date:   2016-01-25
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-25
*/

#ifndef ANDROID_MONKEYVR_COMMON_H
#define ANDROID_MONKEYVR_COMMON_H value

#define LOG_TAG "MonkeyVR"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include <jni.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaMuxer.h>
#include <media/NdkMediaFormat.h>
#include <media/NdkMediaExtractor.h>

#include <stdio.h>

#ifdef ANDROID
    #include <android/log.h>
    #define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
    #define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#else
    #define LOGD(...)
    #define LOGE(...)
#endif

#endif
