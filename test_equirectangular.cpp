/* 
* @Author: BlahGeek
* @Date:   2015-10-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-22
*/

#include <iostream>
#include <gtest/gtest.h>
#include <gtest/extra.h>

#include "./equirectangular.hpp"

using namespace vr;

namespace {

class EquirectangularTest: public ::testing::Test {
public:
    std::unique_ptr<Equirectangular> equir;

    void SetUp() override {
        srand(time(NULL));
        this->equir = std::unique_ptr<Equirectangular>(new Equirectangular(json()));
    }
};

TEST_F(EquirectangularTest, obj_to_image_test) {
    auto lonlat = cv::Point2d(0, 0);
    EXPECT_POINT2_EQ(equir->obj_to_image_single(lonlat),
                     0.5, 0.5, 1e-3);

    lonlat = cv::Point2d(M_PI / 2, M_PI / 4);
    EXPECT_POINT2_EQ(equir->obj_to_image_single(lonlat),
                     0.75, 0.25, 1e-3);
}

TEST_F(EquirectangularTest, reverse_test) {
    for(int i = 0 ; i < 100 ; i += 1) {
        auto lonlat = RAND_LONLAT;
        auto xyz = equir->obj_to_image_single(lonlat);
        EXPECT_POINT2_EQ(equir->image_to_obj_single(xyz), lonlat.x, lonlat.y, 1e-3);
    }
}

}
