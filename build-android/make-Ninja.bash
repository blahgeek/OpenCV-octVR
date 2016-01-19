#!/bin/bash
export ANDROID_NDK="/usr/local/opt/android-ndk/"
export ANDROID_SDK="/usr/local/opt/android-sdk/"
cmake \
    -GNinja \
    -DCMAKE_CXX_FLAGS="-std=c++11" \
    -DCMAKE_TOOLCHAIN_FILE=`pwd`/../platforms/android/android.toolchain.cmake \
    -DWITH_OPENCL=ON \
    -DWITH_CUDA=OFF \
    -DWITH_IPP=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_ANDROID_EXAMPLES=ON \
    -DANDROID_STL=gnustl_static \
    -DANDROID_NATIVE_API_LEVEL=14 \
    -DANDROID_ABI="armeabi-v7a with NEON" \
    -DANDROID_OPENCL_SDK=`pwd`/../android-opencl-sdk \
    -DWITH_TBB=ON \
    ..
