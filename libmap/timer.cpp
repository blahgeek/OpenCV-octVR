/* 
* @Author: BlahGeek
* @Date:   2015-12-07
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-12-07
*/

#include <iostream>
#include "./libmap.hpp"
#include <sys/time.h>

using namespace vr;

static int64_t gettime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

Timer::Timer(std::string name): t(gettime()), name(name) {}
Timer::Timer(): Timer::Timer("") {}

void Timer::tick(std::string msg) {
    auto tn = gettime();
    std::cerr << "[ Timer " << name << "] " << msg << ": " << (tn - t) / 1000.0 << "ms" << std::endl;
    t = tn;
}
