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
__global__ void do_mul_scalar_with_mask(const GlobPtr<T> src,
                                        float scale,
                                        const GlobPtr<uchar> mask,
                                        GlobPtr<T> dst,
                                        int rows, int cols) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < cols && y < rows ) {
            if(mask.row(y)[x] == 0)
                return;

            dst.row(y)[x] = saturate_cast<T>(src.row(y)[x] * scale);
        }
}

void mul_scalar_with_mask(const GpuMat & src,
                          float scale, const GpuMat & mask,
                          GpuMat & dst,
                          cudaStream_t stream) {
    CV_Assert(src.type() == CV_8UC4 || src.type() == CV_8UC3);
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(dst.type() == CV_8UC4 || dst.type() == CV_8UC3);
    CV_Assert(src.type() == dst.type());

    const dim3 block(DefaultTransformPolicy::block_size_x, DefaultTransformPolicy::block_size_y);
    const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

    if(src.type() == CV_8UC3)
        do_mul_scalar_with_mask<<<grid, block, 0, stream>>>(globPtr<uchar3>(src),
                                                            scale,
                                                            globPtr<uchar>(mask),
                                                            globPtr<uchar3>(dst),
                                                            src.rows, src.cols);
    else
        do_mul_scalar_with_mask<<<grid, block, 0, stream>>>(globPtr<uchar4>(src),
                                                            scale,
                                                            globPtr<uchar>(mask),
                                                            globPtr<uchar4>(dst),
                                                            src.rows, src.cols);
    CV_CUDEV_SAFE_CALL( cudaGetLastError() );
    if(stream == 0)
        CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
}


}}} // namespace cv { namespace cuda { namespace cudev {


#endif /* CUDA_DISABLER */
