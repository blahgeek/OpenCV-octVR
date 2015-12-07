AM_CPPFLAGS = -I../json -I../gtest/include $(OPENCV_CFLAGS) -DCV_OPENCL_RUN_VERBOSE
AM_CXXFLAGS = -Wall

include_HEADERS = libmap.hpp
lib_LIBRARIES = libmap.a
bin_PROGRAMS = dumper mapper

LIBS = $(subst -lopencv_ts,,$(OPENCV_LIBS)) -lz

libmap_a_SOURCES = \
	libmap.cpp \
	libmap_async.cpp \
	mapper_template.cpp \
	timer.cpp \
	camera.cpp \
	cameras/equirectangular.cpp \
	cameras/pinhole_cam.cpp \
	cameras/fisheye_cam.cpp \
	cameras/fullframe_fisheye_cam.cpp \
	cameras/normal.cpp 

dumper_SOURCES = dumper.cpp
dumper_LDADD = libmap.a

mapper_SOURCES = mapper.cpp
mapper_LDADD = libmap.a

check_PROGRAMS = testall

testall_SOURCES = \
	tests/test_camera.cpp \
	tests/test_equirectangular.cpp \
	tests/test_pinhole.cpp \
	tests/test.cpp

testall_LDADD = ../gtest/libgtest.a ../gtest/libgtest_main.a libmap.a


TESTS = testall
