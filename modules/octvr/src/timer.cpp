/* 
* @Author: BlahGeek
* @Date:   2015-12-07
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-21
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
#elif defined(__linux__) || defined(__APPLE__)
    #include <sys/time.h>
    static int64_t gettime(void) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
    }
#elif defined(_WIN32)
    static int64_t gettime(void) {
        LARGE_INTEGER tv;
        QueryPerformanceCounter(&tv);
        return tv.QuadPart;
    }
#endif

Timer::Timer(std::string name): t(gettime()), name(name) {
#if defined(_WIN32)
	LARGE_INTEGER fr;
	QueryPerformanceFrequency(&fr);
	frequency = fr.QuadPart;
#endif
}

Timer::Timer(): Timer::Timer("") {}

#ifdef ANDROID
    #include <android/log.h>
    #define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "octVR", __VA_ARGS__))
#endif

void Timer::tick(std::string msg) {
    auto tn = gettime();
	double time_elapsed;

#ifdef ANDROID
    LOGD("[Timer %s] %s: %.2fms", name.c_str(), msg.c_str(), (tn - t) / 1000.0);
#else
    #if defined(_WIN32)
        time_elapsed = (tn - t) * 1000.0 / frequency;
    #else
        time_elapsed = (tn - t) / 1000.0;
    #endif
    std::cerr << "[ Timer " << name << "] " << msg << ": " << time_elapsed << "ms" << std::endl;
#endif

    t = tn;
}
