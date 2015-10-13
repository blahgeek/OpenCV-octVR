/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-13
*/

#include "./equirectangular.hpp"

using namespace vr;

std::pair<double, double> Equirectangular::lonlat_to_xy(double lon, double lat) {
    return std::make_pair<double, double>(lon / (M_PI * 2.0) + 0.5, 
                                          0.5 - lat / M_PI);
}

std::pair<double, double> Equirectangular::xy_to_lonlat(double x, double y) {
    return std::make_pair<double, double>((x - 0.5) * M_PI * 2, (0.5 - y) * M_PI);
}
