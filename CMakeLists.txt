add_library(map STATIC
            libmap.cpp
            libmap_async.cpp
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
add_executable(mapper mapper.cpp)
add_executable(dumper dumper.cpp)

install(TARGETS map ARCHIVE DESTINATION lib)
install(FILES libmap.hpp DESTINATION include)
