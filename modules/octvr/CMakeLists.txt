if (NOT HAVE_QT5)
    message(WARNING "Building octvr without QT support")
else(NOT HAVE_QT5)
    message("Building with QT support")
    find_package(Qt5Core REQUIRED)
    link_libraries(Qt5::Core)
    add_definitions(-DHAVE_QT5)
endif(NOT HAVE_QT5)

#include(FindOpenSSL)

option(WITH_DONGLE_LICENSE "Enable dongle license protector" OFF)
if (WITH_DONGLE_LICENSE)
    find_library(LIBDL doge)
    find_path(LIBDL_INC_DIR "dongle_license.h" PATH_SUFFIXES "doge")
    if (NOT LIBDL OR NOT LIBDL_INC_DIR)
        message(FATAL_ERROR "libdoge not found")
    endif (NOT LIBDL OR NOT LIBDL_INC_DIR)
    #if (NOT OPENSSL_FOUND)
        #message(FATAL_ERROR "OpenSSL not found")
    #endif(NOT OPENSSL_FOUND)
    include_directories(${LIBDL_INC_DIR})
    #include_directories(${OPENSSL_INCLUDE_DIR})
    link_libraries(${LIBDL})
    #link_libraries(${OPENSSL_CRYPTO_LIBRARY})
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        link_libraries("Rpcrt4.lib")
    endif()
    add_definitions(-DWITH_DONGLE_LICENSE)
    message("Dongle license enabled")
else(WITH_DONGLE_LICENSE)
    message("Dongle license disabled")
endif (WITH_DONGLE_LICENSE)

option(WITH_OCTVR_LOGO "Enable hard logo for octVR" OFF)
if (WITH_OCTVR_LOGO)
    message("OctVR Logo Enabled")
    add_definitions(-DWITH_OCTVR_LOGO)
    execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tools/logo_converter.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}/images/logo.png"
                    "${CMAKE_CURRENT_BINARY_DIR}/src/logo_png.cpp")
    
    macro(ocv_get_module_external_sources)
        set(__filesvar "OPENCV_MODULE_opencv_octvr_SOURCES")
        list(APPEND ${__filesvar} "${CMAKE_CURRENT_BINARY_DIR}/src/logo_png.cpp")  
    endmacro(ocv_get_module_external_sources)
    # TODO : can we unset this macro afterwards?
endif (WITH_OCTVR_LOGO)

set(the_description "octVR, you know it")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wnon-virtual-dtor)
ocv_define_module(octvr 
                  opencv_imgproc opencv_features2d opencv_calib3d opencv_objdetect opencv_stitching
                  OPTIONAL opencv_cudaarithm opencv_cudafilters opencv_cudafeatures2d opencv_cudalegacy)

install(FILES "include/octvr.hpp" DESTINATION "${OPENCV_INCLUDE_INSTALL_PATH}/")
install(DIRECTORY "include/rapidjson" DESTINATION "${OPENCV_INCLUDE_INSTALL_PATH}/")
