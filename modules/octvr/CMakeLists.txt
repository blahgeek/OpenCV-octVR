include(FindOpenSSL)

option(WITH_DONGLE_LICENSE "Enable dongle license protector" OFF)
if (WITH_DONGLE_LICENSE)
    find_library(LIBDL DL)
    find_path(LIBDL_INC_DIR "dongle_license.h" PATH_SUFFIXES "doge")
    if (NOT LIBDL OR NOT LIBDL_INC_DIR)
        message(FATAL_ERROR "DL not found")
    endif (NOT LIBDL OR NOT LIBDL_INC_DIR)
    if (NOT OPENSSL_FOUND)
        message(FATAL_ERROR "OpenSSL not found")
    endif(NOT OPENSSL_FOUND)
    include_directories(${LIBDL_INC_DIR})
    include_directories(${OPENSSL_INCLUDE_DIR})
    link_libraries(${LIBDL})
    link_libraries(${OPENSSL_CRYPTO_LIBRARY})
    add_definitions(-DWITH_DONGLE_LICENSE)
    message("Dongle license enabled")
else(${WITH_DONGLE_LICENSE})
    message("Dongle license disabled")
endif (WITH_DONGLE_LICENSE)

set(the_description "octVR, you know it")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wnon-virtual-dtor)
ocv_define_module(octvr 
                  opencv_imgproc opencv_features2d opencv_calib3d opencv_objdetect opencv_stitching
                  OPTIONAL opencv_cudaarithm opencv_cudafilters opencv_cudafeatures2d opencv_cudalegacy)

install(FILES "include/octvr.hpp" DESTINATION "${OPENCV_INCLUDE_INSTALL_PATH}/")
