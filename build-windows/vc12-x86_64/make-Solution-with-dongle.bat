cmake -G "Visual Studio 12 2013 Win64" ^
    -D CMAKE_BUILD_TYPE=Release ^
    -D CMAKE_PREFIX_PATH="C:\Qt\Qt5.5.1\5.5\msvc2013_64\lib\cmake" ^
    -D CMAKE_INSTALL_PREFIX=%cd%\..\..\.. ^
    -D WITH_IPP=OFF ^
    -D WITH_FFMPEG=OFF ^
    -D CMAKE_CXX_FLAGS="-DCV_OPENCL_RUN_VERBOSE" ^
    -D WITH_CUDA=ON ^
    -D WITH_QT=ON ^
    -D CUDA_ARCH_BIN="5.2" ^
    -D CUDA_ARCH_PTX="5.2" ^
    -D BUILD_SHARED_LIBS=ON ^
    -D WITH_OPENCL=ON ^
    -D WITH_OPENEXR=OFF ^
    -D WITH_VFW=OFF ^
    -D BUILD_TESTS=OFF ^
    -D BUILD_PERF_TESTS=OFF ^
    -D BUILD_opencv_python2=OFF ^
    -D BUILD_opencv_python3=OFF ^
    -D WITH_DONGLE_LICENSE=ON ^
    %* ^
    ..\..
