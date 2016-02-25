set QT5_DIR="C:\Qt\Qt5.5.1\5.5\msvc2013_64\lib\cmake"
cmake ^
    -D CMAKE_BUILD_TYPE=Release ^
    -D CMAKE_INSTALL_PREFIX=%cd%\..\.. ^
    -D WITH_FFMPEG=OFF ^
    -D CMAKE_CXX_FLAGS="-DCV_OPENCL_RUN_VERBOSE" ^
    -D WITH_CUDA=ON ^
    -D CUDA_ARCH_BIN="5.2" ^
    -D CUDA_ARCH_PTX="5.2" ^
    -D BUILD_SHARED_LIBS=OFF ^
    -D WITH_OPENCL=ON ^
    -D WITH_OPENEXR=OFF ^
    -D WITH_VFW=OFF ^
    -D BUILD_TESTS=OFF ^
    -D BUILD_PERF_TESTS=OFF ^
    -D Qt5Widgets_DIR="%QT5_DIR%\Qt5Widgets" ^
    -D Qt5Core_DIR="%QT5_DIR%\Qt5Core" ^
    -D Qt5Gui_DIR="%QT5_DIR%\Qt5Gui" ^
    -D Qt5Network_DIR="%QT5_DIR%\Qt5Network" ^
    -D Qt5Multimedia_DIR="%QT5_DIR%\Qt5Multimedia" ^
    -D Qt5MultimediaWidgets_DIR="%QT5_DIR%\Qt5MultimediaWidgets" ^
    ..
