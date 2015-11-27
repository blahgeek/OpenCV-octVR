
#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {

        texture< uchar, cudaTextureType2D> tex_pyr_down (0, cudaFilterModePoint, cudaAddressModeClamp);

        __global__ void fastPyrDown(PtrStepSz<uchar> dst, int dst_cols, int xoff, int yoff)
        {
            typedef float work_t;

            __shared__ work_t smem[256 + 4];

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y;

            const int src_y = 2 * y;

            {
                work_t sum;

                sum =       0.0625f * tex2D(tex_pyr_down, xoff + x, yoff + src_y - 2);
                sum = sum + 0.25f   * tex2D(tex_pyr_down, xoff + x, yoff + src_y - 1);
                sum = sum + 0.375f  * tex2D(tex_pyr_down, xoff + x, yoff + src_y    );
                sum = sum + 0.25f   * tex2D(tex_pyr_down, xoff + x, yoff + src_y + 1);
                sum = sum + 0.0625f * tex2D(tex_pyr_down, xoff + x, yoff + src_y + 2);

                smem[2 + threadIdx.x] = sum;
            }

            if(threadIdx.x < 4) {
                const int real_x = (threadIdx.x < 2) ? (x - 2) : (x + 254);

                work_t sum;

                sum =       0.0625f * tex2D(tex_pyr_down, xoff + real_x, yoff + src_y - 2);
                sum = sum + 0.25f   * tex2D(tex_pyr_down, xoff + real_x, yoff + src_y - 1);
                sum = sum + 0.375f  * tex2D(tex_pyr_down, xoff + real_x, yoff + src_y    );
                sum = sum + 0.25f   * tex2D(tex_pyr_down, xoff + real_x, yoff + src_y + 1);
                sum = sum + 0.0625f * tex2D(tex_pyr_down, xoff + real_x, yoff + src_y + 2);

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


        void fastPyrDown_caller(PtrStepSz<uchar> src, PtrStepSz<uchar> srcWhole,
                                int xoff, int yoff, 
                                PtrStepSz<uchar> dst) {

            const dim3 block(256);
            const dim3 grid(divUp(src.cols, block.x), dst.rows);

            bindTexture(&tex_pyr_down, srcWhole);

            fastPyrDown<<<grid, block>>>(dst, dst.cols, xoff, yoff);
            cudaSafeCall( cudaGetLastError() );
            cudaSafeCall( cudaDeviceSynchronize() );
        }

    } // namespace imgproc
}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
