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

template <typename T>
__global__ void do_vr_add_sub_and_multiply(const GlobPtr<T> a,
                                           const GlobPtr<T> t,
                                           const GlobPtr<float> w,
                                           GlobPtr<short3> d,
                                           const int rows, const int cols) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < cols && y < rows) {
            short3 sub;
            T a_elem = a.row(y)[x];
            T t_elem = t.row(y)[x];
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
template <typename TYPE>
__host__ void vr_add_sub_and_multiply(const GpuMat & A, 
                                      const GpuMat & T, 
                                      const GpuMat & W, 
                                      GpuMat & D, cudaStream_t stream) {
    CV_Assert(A.type() == CV_8UC3 || A.type() == CV_8UC4);
    CV_Assert(T.type() == CV_8UC3 || T.type() == CV_8UC4);
    CV_Assert(W.type() == CV_32F);
    CV_Assert(D.type() == CV_16SC3);
    CV_Assert(A.size() == T.size() && A.size() == W.size() && A.size() == D.size());

    const dim3 block(DefaultTransformPolicy::block_size_x, DefaultTransformPolicy::block_size_y);
    const dim3 grid(divUp(A.cols, block.x), divUp(A.rows, block.y));

    do_vr_add_sub_and_multiply<<<grid, block, 0 stream>>>(globPtr<TYPE>(A),
                                                          globPtr<TYPE>(T),
                                                          globPtr<float>(W),
                                                          globPtr<short3>(D),
                                                          A.rows, A.cols);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

template void vr_add_sub_and_multiply<uchar3>(const GpuMat &, const GpuMat &, const GpuMat &, GpuMat &, cudaStream_t);
template void vr_add_sub_and_multiply<uchar4>(const GpuMat &, const GpuMat &, const GpuMat &, GpuMat &, cudaStream_t);

template <typename T>
__global__ void do_vr_add_multiply(const GlobPtr<T> a,
                                   const GlobPtr<float> w,
                                   GlobPtr<short3> d,
                                   const int rows, const int cols) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < cols && y < rows) {
            short3 sub;
            T a_elem = a.row(y)[x];
            float w_elem = w.row(y)[x];

            sub.x = (a_elem.x) * w_elem;
            sub.y = (a_elem.y) * w_elem;
            sub.z = (a_elem.z) * w_elem;

            short3 * d_p = d.row(y) + x;
            (*d_p).x += sub.x;
            (*d_p).y += sub.y;
            (*d_p).z += sub.z;
        }
}

template <typename TYPE>
__host__ void vr_add_multiply(const GpuMat & A, 
                              const GpuMat & W, 
                              GpuMat & D, cudaStream_t stream) {
    CV_Assert(A.type() == CV_8UC3 || A.type() == CV_8UC4);
    CV_Assert(W.type() == CV_32F);
    CV_Assert(D.type() == CV_16SC3);
    CV_Assert(A.size() == W.size() && A.size() == D.size());

    const dim3 block(DefaultTransformPolicy::block_size_x, DefaultTransformPolicy::block_size_y);
    const dim3 grid(divUp(A.cols, block.x), divUp(A.rows, block.y));

    do_vr_add_multiply<<<grid, block, 0 stream>>>(globPtr<TYPE>(A),
                                                  globPtr<float>(W),
                                                  globPtr<short3>(D),
                                                  A.rows, A.cols);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}

template void vr_add_multiply<uchar3>(const GpuMat &, const GpuMat &, GpuMat &, cudaStream_t);
template void vr_add_multiply<uchar4>(const GpuMat &, const GpuMat &, GpuMat &, cudaStream_t);

}}} // namespace cv { namespace cuda { namespace cudev {


#endif /* CUDA_DISABLER */
