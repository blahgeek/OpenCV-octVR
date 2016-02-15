/* 
* @Author: BlahGeek
* @Date:   2015-12-07
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-24
*/

#include <iostream>
#include "octvr.hpp"


using namespace vr;

#if defined(__linux__) || defined(__APPLE__)
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

void Timer::tick(std::string msg) {
    auto tn = gettime();
	double time_elapsed;
#if defined(_WIN32)
	time_elapsed = (tn - t) * 1000.0 / frequency;
#else
	time_elapsed = (tn - t) / 1000.0;
#endif
    std::cerr << "[ Timer " << name << "] " << msg << ": " << time_elapsed << "ms" << std::endl;
    t = tn;
}
