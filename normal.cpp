/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-13
*/

#include <iostream>
#include "./normal.hpp"

using namespace vr;

Normal::Normal(const json & options): Map(options) {
    this->cam_opt = options["cam_opt"].get<double>();
    this->aspect_ratio = options["aspect_ratio"].get<double>();

    std::cerr << "Normal camera: " << cam_opt << "/" << aspect_ratio << std::endl;
}

std::pair<double, double> Normal::xy_to_lonlat(double x, double y) {
    double cam_x, cam_y, cam_z;
    cam_x = this->cam_opt;
    cam_z = sqrt((1.0 - cam_x * cam_x) / 
                 (1.0 + 1.0 / this->aspect_ratio / this->aspect_ratio));
    cam_y = cam_z / this->aspect_ratio;

    double xx = cam_x;
    double yy = cam_y - y * 2.0 * cam_y;
    double zz = -cam_z + x * 2.0 * cam_z;

    double _sum = sqrt(xx*xx + yy*yy + zz*zz);
    xx /= _sum;
    yy /= _sum;
    zz /= _sum;

    return std::make_pair<double, double>(atan2(zz, xx), asin(yy));
}
