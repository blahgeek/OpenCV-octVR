remove_definitions(-D__OPENCV_BUILD=1)

set(OPENCV_LIVESTITCHING_DEPS opencv_core opencv_highgui opencv_photo opencv_video opencv_objdetect)
ocv_check_dependencies(${OPENCV_LIVESTITCHING_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Multimedia REQUIRED)
find_package(Qt5MultimediaWidgets REQUIRED)

ocv_include_modules_recurse(${OPENCV_LIVESTITCHING_DEPS})
ocv_include_directories("${OpenCV_SOURCE_DIR}/include/opencv")

ocv_add_executable(livestitching
                   main.cpp
                   mainwindow.cpp
                   inputs_select.cpp
                   qjsonitem.cpp
                   qjsonmodel.cpp
                   icons.qrc
                   style.qrc
                   )

ocv_target_link_libraries(livestitching ${OPENCV_LIVESTITCHING_DEPS})
target_link_libraries(livestitching Qt5::Widgets Qt5::Multimedia Qt5::MultimediaWidgets)
