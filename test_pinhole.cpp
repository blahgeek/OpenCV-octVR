/* 
* @Author: BlahGeek
* @Date:   2015-10-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-22
*/

#include <iostream>
#include <gtest/gtest.h>
#include <gtest/extra.h>

#include "./pinhole_cam.hpp"

using namespace vr;

namespace {

class PinholeTest: public ::testing::Test {
protected:
    std::unique_ptr<PinholeCamera> cam;

    void SetUp() override {
        cam = std::unique_ptr<PinholeCamera>(new PinholeCamera(R"(
                {
                    "width": 640,
                    "height": 480,
                    "fx": 500,
                    "fy": 500,
                    "cx": 320,
                    "cy": 240,
                    "dist_coeffs": [0,0,0,0]
                }
                                             )"_json));
    }

    void test_map(cv::Point2d lonlat, double x, double y) {
        auto ret = cam->obj_to_image(std::vector<cv::Point2d>({lonlat}));
        auto xy = ret[0];
        if(isnan(x) || isnan(y))
            EXPECT_TRUE(isnan(xy.x) || isnan(xy.y) || 
                        xy.x < 0 || xy.x > 1 ||
                        xy.y < 0 || xy.y > 1);
        else
            EXPECT_POINT2_EQ(xy, x, y, 1e-3);
    }

};

TEST_F(PinholeTest, map) {
    test_map(cv::Point2d(-M_PI / 2, 0), 0.5, 0.5);
    test_map(cv::Point2d(M_PI / 2, 0), NAN, NAN);
    test_map(cv::Point2d(-M_PI / 2 + atan(0.4), 0), 52.0/64, 0.5);
    test_map(cv::Point2d(-M_PI / 2, atan(0.2)), 0.5, 340.0 / 480);
    test_map(cv::Point2d(-M_PI / 2, -M_PI / 4), NAN, NAN);
}


}
