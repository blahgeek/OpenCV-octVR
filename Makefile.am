AM_CPPFLAGS = -I../ -I../gtest/include
AM_CXXFLAGS = -Wall

lib_LIBRARIES = libmap.a

libmap_a_SOURCES = \
	base.cpp \
	equirectangular.cpp

check_PROGRAMS = testall

testall_SOURCES = test.cpp

testall_LDADD = ../gtest/libgtest.a ../gtest/libgtest_main.a


TESTS = testall
