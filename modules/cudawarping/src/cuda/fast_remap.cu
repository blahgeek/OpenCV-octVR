
#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/cudev/ptr2d/glob.hpp"
#include "opencv2/cudev/ptr2d/texture.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/filters.hpp"
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/cudev/ptr2d/glob.hpp"
#include "opencv2/cudev/ptr2d/texture.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        template <typename T>
        __global__ void fast_remap(cv::cudev::Texture<T> src,
                                   PtrStepf mapx, PtrStepf mapy,
                                   PtrStepSz<T> dst,
                                   bool fill_zero) {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                const float xcoo = mapx.ptr(y)[x];
                const float ycoo = mapy.ptr(y)[x];
                if(xcoo < 0) {
                    if(fill_zero)
                        dst.ptr(y)[x] = VecTraits<T>::all(0);
                    return;
                }

                typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;
                work_type val = tex2D<work_type>(src.texObj, xcoo, ycoo);
                /*work_type val = src(ycoo, xcoo);*/

                dst.ptr(y)[x] = saturate_cast<T>(val * static_cast<float>(numeric_limits<typename VecTraits<T>::elem_type>::max()));
            }
        }

        template <typename T>
        void fast_remap_caller(GpuMat src, PtrStepf mapx, PtrStepf mapy, PtrStepSz<T> dst, bool fill_zero, cudaStream_t stream) {
            cv::cudev::Texture<T> src_tex(cv::cudev::globPtr<T>(src), true, cudaFilterModeLinear);

            dim3 block(16, 16);
            dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

            fast_remap<<<grid, block, 0, stream>>>(src_tex, mapx, mapy, dst, fill_zero);
            cudaSafeCall( cudaGetLastError() );
            if(stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void fast_remap_caller(GpuMat, PtrStepf, PtrStepf, PtrStepSz<uchar4>, bool fill_zero, cudaStream_t);
        
    } // namespace imgproc
}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
