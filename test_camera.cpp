/* 
* @Author: BlahGeek
* @Date:   2015-10-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-22
*/

#include <iostream>
#include <gtest/gtest.h>
#include <gtest/extra.h>

#include <time.h>
#include <stdlib.h>

#include "./camera.hpp"

using namespace vr;

namespace {

class CameraTest: public ::testing::Test, public Camera {
public:
    CameraTest(): Camera(json()) {}
};

TEST_F(CameraTest, sample_test) {
    EXPECT_EQ(this->get_aspect_ratio(), 1.0);
}

TEST_F(CameraTest, sphere_transform_test) {
    auto lonlat = cv::Point2d(0, 0);
    EXPECT_POINT3_EQ(sphere_lonlat_to_xyz(lonlat), 1, 0, 0, 1e-3);

    lonlat = cv::Point2d(M_PI / 2, M_PI / 4);
    EXPECT_POINT3_EQ(sphere_lonlat_to_xyz(lonlat), 0, 0.707106, -0.707106, 1e-3);

    lonlat = cv::Point2d(-M_PI / 4 * 3, -M_PI / 4);
    EXPECT_POINT3_EQ(sphere_lonlat_to_xyz(lonlat), -0.5, -0.707106, 0.5, 1e-3);

    lonlat = cv::Point2d(-M_PI / 4, -M_PI / 2);
    EXPECT_POINT3_EQ(sphere_lonlat_to_xyz(lonlat), 0, -1, 0, 1e-3);

    srand(time(NULL));
    for(int i = 0 ; i < 100 ; i += 1) {
        lonlat = RAND_LONLAT;
        auto xyz = sphere_lonlat_to_xyz(lonlat);
        EXPECT_POINT2_EQ(sphere_xyz_to_lonlat(xyz), lonlat.x, lonlat.y, 1e-3);
    }
}

// TODO: test sphere_rotate()

}
