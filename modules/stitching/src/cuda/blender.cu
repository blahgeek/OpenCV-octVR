#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/cudev/ptr2d/glob.hpp"
#include "opencv2/cudev/grid/transform.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

using namespace cv::cudev;

namespace cv { namespace cuda { namespace device {

__global__ void do_vr_add_sub_and_multiply(const GlobPtr<uchar3> a,
                                           const GlobPtr<uchar3> t,
                                           const GlobPtr<float> w,
                                           GlobPtr<short3> d,
                                           const int rows, const int cols) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < cols && y < rows) {
            short3 sub;
            uchar3 a_elem = a.row(y)[x];
            uchar3 t_elem = t.row(y)[x];
            float w_elem = w.row(y)[x];

            sub.x = (a_elem.x - t_elem.x) * w_elem;
            sub.y = (a_elem.y - t_elem.y) * w_elem;
            sub.z = (a_elem.z - t_elem.z) * w_elem;

            short3 * d_p = d.row(y) + x;
            (*d_p).x += sub.x;
            (*d_p).y += sub.y;
            (*d_p).z += sub.z;
        }
}

// used by MultiBandGPUBlender
// D += (A - T) * W
__host__ void vr_add_sub_and_multiply(const GpuMat & A, 
                                      const GpuMat & T, 
                                      const GpuMat & W, 
                                      GpuMat & D) {
    CV_Assert(A.type() == CV_8UC3);
    CV_Assert(T.type() == CV_8UC3);
    CV_Assert(W.type() == CV_32F);
    CV_Assert(D.type() == CV_16SC3);
    CV_Assert(A.size() == T.size() && A.size() == W.size() && A.size() == D.size());

    const dim3 block(DefaultTransformPolicy::block_size_x, DefaultTransformPolicy::block_size_y);
    const dim3 grid(divUp(A.cols, block.x), divUp(A.rows, block.y));

    do_vr_add_sub_and_multiply<<<grid, block>>>(globPtr<uchar3>(A),
                                                globPtr<uchar3>(T),
                                                globPtr<float>(W),
                                                globPtr<short3>(D),
                                                A.rows, A.cols);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

__global__ void do_vr_add_sub_and_multiply_channel(const GlobPtr<uchar> a,
                                                   const GlobPtr<uchar> t,
                                                   const GlobPtr<float> w,
                                                   GlobPtr<short3> d,
                                                   const int rows, const int cols,
                                                   const int channel) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < cols && y < rows) {
            uchar a_elem = a.row(y)[x];
            uchar t_elem = t.row(y)[x];
            float w_elem = w.row(y)[x];

            short sub = (a_elem - t_elem) * w_elem;

            switch(channel) {
                case 0: d.row(y)[x].x += sub; break;
                case 1: d.row(y)[x].y += sub; break;
                case 2: d.row(y)[x].z += sub; break;
                default: break;
            }
        }
}

__host__ void vr_add_sub_and_multiply_channel(const GpuMat & A, 
                                              const GpuMat & T, 
                                              const GpuMat & W, 
                                              GpuMat & D,
                                              int channel) {
    CV_Assert(A.type() == CV_8U);
    CV_Assert(T.type() == CV_8U);
    CV_Assert(W.type() == CV_32F);
    CV_Assert(D.type() == CV_16SC3);
    CV_Assert(A.size() == T.size() && A.size() == W.size() && A.size() == D.size());

    const dim3 block(DefaultTransformPolicy::block_size_x, DefaultTransformPolicy::block_size_y);
    const dim3 grid(divUp(A.cols, block.x), divUp(A.rows, block.y));

    do_vr_add_sub_and_multiply_channel<<<grid, block>>>(globPtr<uchar>(A),
                                                        globPtr<uchar>(T),
                                                        globPtr<float>(W),
                                                        globPtr<short3>(D),
                                                        A.rows, A.cols, channel);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

}}} // namespace cv { namespace cuda { namespace cudev {


#endif /* CUDA_DISABLER */
