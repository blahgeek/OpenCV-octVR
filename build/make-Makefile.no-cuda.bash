#!/bin/bash
# GTX 960, see https://developer.nvidia.com/cuda-gpus
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=`pwd`/../.. \
    -D WITH_FFMPEG=OFF \
    -D CMAKE_CXX_FLAGS="-std=c++11 -DCV_OPENCL_RUN_VERBOSE" \
    -D WITH_CUDA=OFF \
    -D BUILD_SHARED_LIBS=ON \
    -D WITH_OPENCL=ON \
    -D WITH_OPENEXR=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    $@ \
    ..
