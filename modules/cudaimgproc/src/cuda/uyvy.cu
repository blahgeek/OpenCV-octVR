#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/cudev/ptr2d/glob.hpp"
#include "opencv2/cudev/grid/transform.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"

using namespace cv::cudev;

namespace cv { namespace cuda { namespace device
{

    template <typename T>
    __global__ void splitUYVYKernel(int u_rows, int u_cols,
                                    const GlobPtr<T> uyvy,
                                    GlobPtr<T> _y, GlobPtr<T> _u, GlobPtr<T> _v) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(y < u_rows && x < u_cols) {
            const T * src_p = uyvy.ptr(y) + x * 4;
            _u.ptr(y)[x]         = src_p[0];
            _y.ptr(y)[x * 2]     = src_p[1];
            _v.ptr(y)[x]         = src_p[2];
            _y.ptr(y)[x * 2 + 1] = src_p[3];
        }
    }

    __host__ void splitUYVYCaller(const GpuMat & _uyvy,
                                  GpuMat & y, GpuMat & u, GpuMat & v, 
                                  cudaStream_t stream) {
        GpuMat uyvy = _uyvy.reshape(1);
        CV_Assert(uyvy.type() == CV_8U);

        y.create(uyvy.rows, uyvy.cols / 2, CV_8U);
        u.create(uyvy.rows, uyvy.cols / 4, CV_8U);
        v.create(uyvy.rows, uyvy.cols / 4, CV_8U);

        const dim3 block(DefaultTransformPolicy::block_size_x, DefaultTransformPolicy::block_size_y);
        const dim3 grid(divUp(u.cols, block.x), divUp(u.rows, block.y));

        splitUYVYKernel<<<grid, block, 0, stream>>>(u.rows, u.cols, globPtr<uchar>(uyvy),
                                                    globPtr<uchar>(y), globPtr<uchar>(u), globPtr<uchar>(v));
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );
        if(stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template <typename T>
    __global__ void mergeUYVYKernel(int u_rows, int u_cols,
                                    const GlobPtr<T> _y, 
                                    const GlobPtr<T> _u, 
                                    const GlobPtr<T> _v,
                                    GlobPtr<T> uyvy) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(y < u_rows && x < u_cols) {
            T * dst_p = uyvy.ptr(y) + x * 4;
            dst_p[0] = _u.ptr(y)[x];
            dst_p[1] = _y.ptr(y)[x * 2];
            dst_p[2] = _v.ptr(y)[x];
            dst_p[3] = _y.ptr(y)[x * 2 + 1];
        }
    }

    __host__ void mergeUYVYCaller(const GpuMat & _y, GpuMat & _u, GpuMat & _v,
                                  GpuMat & uyvy,
                                  cudaStream_t stream) {
        GpuMat y = _y.reshape(1);
        CV_Assert(y.type() == CV_8U);

        GpuMat u = _u.reshape(1);
        CV_Assert(u.type() == CV_8U);
        CV_Assert(u.rows == y.rows && u.cols * 2 == y.cols);

        GpuMat v = _v.reshape(1);
        CV_Assert(v.type() == CV_8U);
        CV_Assert(v.rows == y.rows && v.cols * 2 == y.cols);

        uyvy.create(u.rows, u.cols * 4, CV_8U);

        const dim3 block(DefaultTransformPolicy::block_size_x, DefaultTransformPolicy::block_size_y);
        const dim3 grid(divUp(u.cols, block.x), divUp(u.rows, block.y));

        mergeUYVYKernel<<<grid, block, 0, stream>>>(u.rows, u.cols, 
                                                    globPtr<uchar>(y), globPtr<uchar>(u), globPtr<uchar>(v),
                                                    globPtr<uchar>(uyvy));
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );
        if(stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
