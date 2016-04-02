/* 
* @Author: StrayWarrior
* @Date:   2016-04-01
* @Last Modified by:   StrayWarrior 
* @Last Modified time: 2016-04-02
*/

#ifndef VR_PARALLEL_CALLER_H
#define VR_PARALLEL_CALLER_H value

#include <opencv2/core/utility.hpp>

namespace vr {

template <typename Body>
class ParallelFunctionCaller : public cv::ParallelLoopBody {
private:
    const Body & _func;
public:
    ParallelFunctionCaller(const Body& _parallel_func) : _func(_parallel_func) {}
    void operator() (const cv::Range& _range) const
    {
        _func(_range);
    }
};

template <typename Body>
void parallel_for_caller(const cv::Range& _range, const Body& _body) {
    cv::parallel_for_(_range, ParallelFunctionCaller<Body>(_body));
}

}

#endif /* VR_PARALLEL_CALLER_H */
