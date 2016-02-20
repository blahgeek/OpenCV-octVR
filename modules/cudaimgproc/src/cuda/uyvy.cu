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
            const T * src_p = uyvy.row(y) + x * 4;
            _u.row(y)[x]         = src_p[0];
            _y.row(y)[x * 2]     = src_p[1];
            _v.row(y)[x]         = src_p[2];
            _y.row(y)[x * 2 + 1] = src_p[3];
        }
    }

    __host__ void splitUYVYCaller(const GpuMat & uyvy,
                                  GpuMat & y, GpuMat & u, GpuMat & v, 
                                  cudaStream_t stream) {

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
            T * dst_p = uyvy.row(y) + x * 4;
            dst_p[0] = _u.row(y)[x];
            dst_p[1] = _y.row(y)[x * 2];
            dst_p[2] = _v.row(y)[x];
            dst_p[3] = _y.row(y)[x * 2 + 1];
        }
    }

    __host__ void mergeUYVYCaller(const GpuMat & y, const GpuMat & u, const GpuMat & v,
                                  GpuMat & uyvy,
                                  cudaStream_t stream) {
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
