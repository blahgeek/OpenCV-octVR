#!/bin/bash
# GTX 960, see https://developer.nvidia.com/cuda-gpus
cmake \
    -D Qt5Core_DIR=/usr/local/opt/qt/lib/cmake/Qt5Core/ \
    -D Qt5Gui_DIR=/usr/local/opt/qt/lib/cmake/Qt5Gui/ \
    -D Qt5Concurrent_DIR=/usr/local/opt/qt/lib/cmake/Qt5Concurrent/ \
    -D Qt5Widgets_DIR=/usr/local/opt/qt/lib/cmake/Qt5Widgets/ \
    -D Qt5Test_DIR=/usr/local/opt/qt/lib/cmake/Qt5Test/ \
    -D Qt5Multimedia_DIR=/usr/local/opt/qt/lib/cmake/Qt5Multimedia/ \
    -D Qt5MultimediaWidgets_DIR=/usr/local/opt/qt/lib/cmake/Qt5MultimediaWidgets/ \
    -D CMAKE_BUILD_TYPE=Debug \
    -D CMAKE_INSTALL_PREFIX=`pwd`/../.. \
    -D WITH_IPP=OFF \
    -D WITH_TBB=OFF \
    -D WITH_FFMPEG=OFF \
    -D CMAKE_CXX_FLAGS="-std=c++11 -DCV_OPENCL_RUN_VERBOSE" \
    -D WITH_CUDA=OFF \
    -D WITH_QT=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D WITH_OPENCL=OFF \
    -D WITH_QUICKTIME=OFF \
    -D WITH_OPENEXR=OFF \
    -D WITH_VFW=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D WITH_OCTVR_LOGO=ON \
    $@ \
    ..
