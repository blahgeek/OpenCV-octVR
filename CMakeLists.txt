add_library(map STATIC
            libmap.cpp
            libmap_loaddump.cpp
            camera.cpp
            equirectangular.cpp
            pinhole_cam.cpp
            fisheye_cam.cpp
            fullframe_fisheye_cam.cpp
            normal.cpp 
           )

link_libraries(map)
add_executable(mapper mapper.cpp)
add_executable(dumper dumper.cpp)
