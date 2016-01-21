#!/bin/bash
# GTX 960, see https://developer.nvidia.com/cuda-gpus
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=`pwd`/../.. \
    -D WITH_FFMPEG=OFF \
    -D CMAKE_CXX_FLAGS="-std=c++11 -DCV_OPENCL_RUN_VERBOSE" \
    -D CUDA_NVCC_FLAGS="-std=c++11 --expt-relaxed-constexpr" \
    -D WITH_CUDA=ON \
    -D BUILD_SHARED_LIBS=OFF \
    -D CUDA_ARCH_BIN="5.2" \
    -D CUDA_ARCH_PTX="5.2" \
    -D WITH_OPENCL=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    ..
