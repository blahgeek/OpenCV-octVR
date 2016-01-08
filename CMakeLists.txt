add_library(map STATIC
            libmap.cpp
            mapper_template.cpp
            timer.cpp
            camera.cpp
            cameras/equirectangular.cpp
            cameras/pinhole_cam.cpp
            cameras/fisheye_cam.cpp
            cameras/fullframe_fisheye_cam.cpp
            cameras/normal.cpp
           )

link_libraries(map)
add_executable(dumper dumper.cpp)
add_executable(monkey_gen monkey_gen.cpp)
add_executable(mapper mapper.cpp)

install(TARGETS map ARCHIVE DESTINATION lib)
install(FILES libmap.hpp DESTINATION include)

if(WITH_TESTS)
    add_executable(testall 
                   tests/test_camera.cpp
                   tests/test_equirectangular.cpp
                   tests/test_pinhole.cpp
                   tests/test.cpp
                  )
    target_link_libraries(testall gtest gtest_main)
    add_test(NAME testall COMMAND testall)
endif(WITH_TESTS)
