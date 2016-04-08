/* 
* @Author: StrayWarrior
* @Date:   2016-04-01
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-08
*/

#ifndef VR_PARALLEL_CALLER_H
#define VR_PARALLEL_CALLER_H value

#include <cmath>
#include <algorithm>
#include <opencv2/core/utility.hpp>

namespace vr {

template <typename Body>
class ParallelFunctionCaller : public cv::ParallelLoopBody {
private:
    const Body & _func;
    int block_size;
    cv::Range elem_range;
public:
    ParallelFunctionCaller(const Body& _parallel_func, int _bs, cv::Range _r) : 
        _func(_parallel_func), block_size(_bs), elem_range(_r) {}
    void operator() (const cv::Range& _range) const
    {
        cv::Range real_range(std::max(_range.start * block_size, elem_range.start), 
                             std::min(_range.end * block_size, elem_range.end));
        _func(real_range);
    }
};

template <typename Body>
void parallel_for_caller(const cv::Range& _range, const Body& _body, int block_size=64) {
    cv::Range _block_range(std::floor(_range.start / float(block_size)),
                           std::ceil(_range.end / float(block_size)));
    cv::parallel_for_(_block_range, ParallelFunctionCaller<Body>(_body, block_size, _range));
}

}

#endif /* VR_PARALLEL_CALLER_H */
