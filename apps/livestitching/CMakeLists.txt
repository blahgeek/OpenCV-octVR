remove_definitions(-D__OPENCV_BUILD=1)
unset(LIBSODIUM)

set(OPENCV_LIVESTITCHING_DEPS 
    opencv_core opencv_highgui opencv_photo opencv_video opencv_objdetect opencv_octvr)
ocv_check_dependencies(${OPENCV_LIVESTITCHING_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

if (NOT HAVE_QT5)
  message("Livestitching is not building")
  return()
endif(NOT HAVE_QT5)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Multimedia REQUIRED)
find_package(Qt5MultimediaWidgets REQUIRED)

ocv_include_modules_recurse(${OPENCV_LIVESTITCHING_DEPS})
ocv_include_directories("${OpenCV_SOURCE_DIR}/include/opencv")

option(OWLLIVE_ENCRYPT_ARG "Encrypt the arguments passed by OwlLive" OFF)
if (OWLLIVE_ENCRYPT_ARG)
    # For safety, it must link to a static library.
    if(WIN32)
        find_library(LIBSODIUM libsodium
                     NAMES libsodium.lib 
                     PATHS ENV LIBPATH)
    else()
        find_library(LIBSODIUM libsodium
                     NAMES libsodium.a)
    endif()
    message(${LIBSODIUM})
    try_compile(HAVE_LIBSODIUM
                "${OpenCV_BINARY_DIR}"
                "${OpenCV_SOURCE_DIR}/cmake/checks/libsodium.cpp"
                LINK_LIBRARIES ${LIBSODIUM})
    if (NOT HAVE_LIBSODIUM)
        message(FATAL_ERROR "libsodium not found")
    endif()
    add_definitions(-DOWLLIVE_ENCRYPT_ARG -DSODIUM_STATIC)
    message("OwlLive arguments will be encrypted")
else(OWLLIVE_ENCRYPT_ARG)
    message("OwlLive arguments will not be encrypted")
endif(OWLLIVE_ENCRYPT_ARG)

option(OWLLIVE_DISABLE_CONSOLE "Disable windows console output" OFF)
if (OWLLIVE_DISABLE_CONSOLE)
    add_definitions(-DOWLLIVE_DISABLE_CONSOLE)
    add_definitions(-DQT_NO_DEBUG_OUTPUT)
    ocv_add_executable(livestitching WIN32
                       main.cpp
                       mainwindow.cpp
                       inputs_select.cpp
                       preview_video.cpp
                       pto_template.cpp
                       runner.cpp
                       qjsonitem.cpp
                       qjsonmodel.cpp
                       encryptor.cpp
                       projection_modes.cpp
                       icons.qrc
                       style.qrc
                       )
else(OWLLIVE_DISABLE_CONSOLE)
    ocv_add_executable(livestitching
                       main.cpp
                       mainwindow.cpp
                       inputs_select.cpp
                       preview_video.cpp
                       pto_template.cpp
                       runner.cpp
                       qjsonitem.cpp
                       qjsonmodel.cpp
                       encryptor.cpp
                       projection_modes.cpp
                       icons.qrc
                       style.qrc
                       )
endif(OWLLIVE_DISABLE_CONSOLE)


if (OWLLIVE_ENCRYPT_ARG)
    ocv_target_link_libraries(livestitching ${LIBSODIUM})
    #ocv_target_include_directories(livestitching ${LIBSODIUM_INC_DIR})
endif(OWLLIVE_ENCRYPT_ARG)

ocv_target_link_libraries(livestitching ${OPENCV_LIVESTITCHING_DEPS} ${QT_QTMAIN_LIBRARY})
target_link_libraries(livestitching Qt5::Widgets Qt5::Multimedia Qt5::MultimediaWidgets)
