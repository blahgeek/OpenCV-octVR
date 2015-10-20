AM_CPPFLAGS = -I../json -I../gtest/include
AM_CXXFLAGS = -Wall

include_HEADERS = libmap.hpp
lib_LIBRARIES = libmap.a

libmap_a_SOURCES = \
	libmap.cpp \
	camera.cpp \
	equirectangular.cpp \
	pinhole_cam.cpp \
	fisheye_cam.cpp \
	normal.cpp 

check_PROGRAMS = testall

testall_SOURCES = test.cpp

testall_LDADD = ../gtest/libgtest.a ../gtest/libgtest_main.a


TESTS = testall
