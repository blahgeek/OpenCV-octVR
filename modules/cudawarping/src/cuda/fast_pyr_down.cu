
#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/cudev/ptr2d/glob.hpp"
#include "opencv2/cudev/ptr2d/texture.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {

        __global__ void fastPyrDown(cv::cudev::Texture<uchar> src, PtrStepSz<uchar> dst, int dst_cols)
        {
            typedef float work_t;

            __shared__ work_t smem[256 + 4];

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y;

            const int src_y = 2 * y;

            {
                work_t sum;

                sum =       0.0625f * src(src_y - 2, x);
                sum = sum + 0.25f   * src(src_y - 1, x);
                sum = sum + 0.375f  * src(src_y    , x);
                sum = sum + 0.25f   * src(src_y + 1, x);
                sum = sum + 0.0625f * src(src_y + 2, x);

                smem[2 + threadIdx.x] = sum;
            }

            if(threadIdx.x < 4) {
                const int real_x = (threadIdx.x < 2) ? (x - 2) : (x + 254);

                work_t sum;

                sum =       0.0625f * src(src_y - 2, real_x);
                sum = sum + 0.25f   * src(src_y - 1, real_x);
                sum = sum + 0.375f  * src(src_y    , real_x);
                sum = sum + 0.25f   * src(src_y + 1, real_x);
                sum = sum + 0.0625f * src(src_y + 2, real_x);

                const int store_idx = (threadIdx.x < 2 ) ? (threadIdx.x) : (threadIdx.x+256);
                smem[store_idx] = sum;
            }


            __syncthreads();

            if (threadIdx.x < 128)
            {
                const int tid2 = threadIdx.x * 2;

                work_t sum;

                sum =       0.0625f * smem[2 + tid2 - 2];
                sum = sum + 0.25f   * smem[2 + tid2 - 1];
                sum = sum + 0.375f  * smem[2 + tid2    ];
                sum = sum + 0.25f   * smem[2 + tid2 + 1];
                sum = sum + 0.0625f * smem[2 + tid2 + 2];

                const int dst_x = (blockIdx.x * blockDim.x + tid2) / 2;

                if (dst_x < dst_cols)
                    dst.ptr(y)[dst_x] = saturate_cast<uchar>(sum);
            }
        }


        void fastPyrDown_caller(GpuMat src, 
                                GpuMat dst) {

            const dim3 block(256);
            const dim3 grid(divUp(src.cols, block.x), dst.rows);

            cv::cudev::Texture<uchar> src_tex(cv::cudev::globPtr<uchar>(src));

            fastPyrDown<<<grid, block>>>(src_tex, dst, dst.cols);
            cudaSafeCall( cudaGetLastError() );
            cudaSafeCall( cudaDeviceSynchronize() );
        }

    } // namespace imgproc
}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
