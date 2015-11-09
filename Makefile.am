AM_CPPFLAGS = -I../json -I../gtest/include $(OPENCV_CFLAGS)
AM_CXXFLAGS = -Wall

include_HEADERS = libmap.hpp
lib_LIBRARIES = libmap.a
bin_PROGRAMS = dumper mapper

libmap_a_SOURCES = \
	libmap.cpp \
	libmap_loaddump.cpp \
	camera.cpp \
	equirectangular.cpp \
	pinhole_cam.cpp \
	fisheye_cam.cpp \
	fullframe_fisheye_cam.cpp \
	normal.cpp 

dumper_SOURCES = dumper.cpp
dumper_LDADD = libmap.a
dumper_LDFLAGS = $(OPENCV_LIBS)

mapper_SOURCES = mapper.cpp
mapper_LDADD = libmap.a
mapper_LDFLAGS = $(OPENCV_LIBS)

check_PROGRAMS = testall

testall_SOURCES = \
	test_camera.cpp \
	test_equirectangular.cpp \
	test_pinhole.cpp \
	test.cpp

testall_LDADD = ../gtest/libgtest.a ../gtest/libgtest_main.a libmap.a
testall_LDFLAGS = $(subst -lopencv_ts,,$(OPENCV_LIBS))


TESTS = testall
