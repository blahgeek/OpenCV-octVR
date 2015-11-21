#!/bin/bash
# GTX 960, see https://developer.nvidia.com/cuda-gpus
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=`pwd`/../.. \
    -D WITH_FFMPEG=OFF \
    -D CMAKE_CXX_FLAGS="-DCV_OPENCL_RUN_VERBOSE" \
    -D WITH_CUDA=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCUDA_ARCH_BIN="5.2" \
    -DCUDA_ARCH_PTX="5.2" \
    -DWITH_OPENCL=OFF \
    ..
