/* 
* @Author: BlahGeek
* @Date:   2015-12-07
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-22
*/

#include <iostream>

#include "octvr.hpp"

using namespace vr;

#ifdef ANDROID
    #include <time.h>
    static int64_t gettime(void) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        return (int64_t)now.tv_sec * 1000000 + now.tv_nsec / 1000;
    }
#else
    #include <sys/time.h>
    static int64_t gettime(void) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
    }
#endif

Timer::Timer(std::string name): t(gettime()), name(name) {}
Timer::Timer(): Timer::Timer("") {}

#ifdef ANDROID
    #include <android/log.h>
    #define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "octVR", __VA_ARGS__))
#endif

void Timer::tick(std::string msg) {
    auto tn = gettime();
#ifdef ANDROID
    LOGD("[Timer %s] %s: %.2fms", name.c_str(), msg.c_str(), (tn - t) / 1000.0);
#else
    std::cerr << "[ Timer " << name << "] " << msg << ": " << (tn - t) / 1000.0 << "ms" << std::endl;
#endif
    t = tn;
}
