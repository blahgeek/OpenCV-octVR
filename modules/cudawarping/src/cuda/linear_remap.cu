// Faster remap using CUDA texture

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/limits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/filters.hpp"

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        template <typename Ptr2D, typename T> 
        __global__ void linear_remap(const Ptr2D src, const PtrStepf mapx, const PtrStepf mapy, PtrStepSz<T> dst)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                const float xcoo = mapx.ptr(y)[x];
                const float ycoo = mapy.ptr(y)[x];
                if(xcoo < 0)
                    return;

                dst.ptr(y)[x] = saturate_cast<T>(src(ycoo, xcoo));
            }
        }

        template <typename T> struct LinearRemapDispatcher {
            static void call(PtrStepSz<T> src, PtrStepSzf mapx, PtrStepSzf mapy,
                             PtrStepSz<T> dst, cudaStream_t stream) {}
        };

        #define OPENCV_CUDA_IMPLEMENT_REMAP_TEX(type) \
            texture< type , cudaTextureType2D, cudaReadModeNormalizedFloat> \
                tex_linear_remap_ ## type ## (0, cudaFilterModeLinear, cudaAddressModeClamp); \
            struct tex_linear_remap_ ## type ## _reader \
            { \
                typedef type elem_type; \
                typedef float index_type; \
                typedef TypeVec<float, VecTraits<type>::cn>::vec_type return_type; \
                __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const \
                { \
                    return_type ret = tex2D(tex_linear_remap_ ## type , x, y); \
                    ret *= numeric_limits<elem_type>::max(); \
                    return saturate_cast<elem_type>(ret); \
                } \
            }; \
            template <> struct LinearRemapDispatcher< type > \
            { \
                static void call(PtrStepSz< type > src, PtrStepSzf mapx, PtrStepSzf mapy, \
                    PtrStepSz< type > dst, cudaStream_t stream) \
                { \
                    dim3 block(32, 8); \
                    dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
                    bindTexture(&tex_linear_remap_ ## type , src); \
                    tex_linear_remap_ ## type ##_reader texSrc(); \
                    linear_remap<<<grid, block, 0, stream>>>(texSrc, mapx, mapy, dst); \
                    cudaSafeCall( cudaGetLastError() ); \
                } \
            };

        OPENCV_CUDA_IMPLEMENT_REMAP_TEX(uchar)
        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(uchar2)
        OPENCV_CUDA_IMPLEMENT_REMAP_TEX(uchar4)

        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(schar)
        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(char2)
        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(char4)

        OPENCV_CUDA_IMPLEMENT_REMAP_TEX(ushort)
        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(ushort2)
        OPENCV_CUDA_IMPLEMENT_REMAP_TEX(ushort4)

        OPENCV_CUDA_IMPLEMENT_REMAP_TEX(short)
        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(short2)
        OPENCV_CUDA_IMPLEMENT_REMAP_TEX(short4)

        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(int)
        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(int2)
        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(int4)

        OPENCV_CUDA_IMPLEMENT_REMAP_TEX(float)
        //OPENCV_CUDA_IMPLEMENT_REMAP_TEX(float2)
        OPENCV_CUDA_IMPLEMENT_REMAP_TEX(float4)

        #undef OPENCV_CUDA_IMPLEMENT_REMAP_TEX

        template <typename T> void linear_remap_gpu(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap,
            PtrStepSzb dst, cudaStream_t stream)
        {
            CV_Assert(stream != 0);
            LinearRemapDispatcher<T>::call(static_cast<PtrStepSz<T> > (src),
                                           xmap, ymap, 
                                           static_cast<PtrStepSz<T> > (dst),
                                           stream);
        }

        template void linear_remap_gpu<uchar >(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<uchar2>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        template void linear_remap_gpu<uchar3>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        template void linear_remap_gpu<uchar4>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);

        //template void linear_remap_gpu<schar>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<char2>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<char3>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<char4>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);

        template void linear_remap_gpu<ushort >(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<ushort2>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        template void linear_remap_gpu<ushort3>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        template void linear_remap_gpu<ushort4>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);

        template void linear_remap_gpu<short >(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<short2>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        template void linear_remap_gpu<short3>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        template void linear_remap_gpu<short4>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);

        //template void linear_remap_gpu<int >(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<int2>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<int3>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<int4>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);

        template void linear_remap_gpu<float >(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        //template void linear_remap_gpu<float2>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        template void linear_remap_gpu<float3>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
        template void linear_remap_gpu<float4>(PtrStepSzb src, PtrStepSzf xmap, PtrStepSzf ymap, PtrStepSzb dst, cudaStream_t stream);
    } // namespace imgproc
}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
