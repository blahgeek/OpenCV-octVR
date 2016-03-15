ocv_check_dependencies(opencv_octvr)

if(NOT OCV_DEPENDENCIES_FOUND)
   return()
endif()

ocv_include_modules_recurse(opencv_octvr)

ocv_add_executable(octvr_dump dump.cpp)
ocv_add_executable(octvr_map map.cpp)
ocv_add_executable(octvr_monkeygen monkey_gen.cpp)
ocv_add_executable(octvr_gain_and_remap gain_and_remap.cpp)

ocv_target_link_libraries(octvr_dump opencv_octvr)
ocv_target_link_libraries(octvr_map opencv_octvr)
ocv_target_link_libraries(octvr_monkeygen opencv_octvr)
ocv_target_link_libraries(octvr_gain_and_remap opencv_octvr)
