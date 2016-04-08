#!/bin/bash
# GTX 960, see https://developer.nvidia.com/cuda-gpus
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=`pwd`/../.. \
    -D WITH_IPP=OFF \
    -D WITH_TBB=OFF \
    -D WITH_FFMPEG=OFF \
    -D CMAKE_CXX_FLAGS="-std=c++11 -DCV_OPENCL_RUN_VERBOSE" \
    -D CUDA_NVCC_FLAGS="-std=c++11 --expt-relaxed-constexpr" \
    -D WITH_CUDA=ON \
    -D WITH_QT=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D CUDA_ARCH_BIN="5.2" \
    -D CUDA_ARCH_PTX="5.2" \
    -D WITH_OPENCL=ON \
    -D WITH_OPENEXR=OFF \
    -D WITH_VFW=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D WITH_DONGLE_LICENSE=OFF \
    -D OWLLIVE_ENCRYPT_ARG=OFF \
    -D OWLLIVE_DISABLE_CONSOLE=OFF \
    $@ \
    ..
