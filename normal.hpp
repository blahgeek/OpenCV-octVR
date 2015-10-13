/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-13
*/

#ifndef VR_LIBMAP_NORMAL_H
#define VR_LIBMAP_NORMAL_H value

#include "./libmap.hpp"

namespace vr {

class Normal: public Map {
private:
    double cam_opt;
    double aspect_ratio;
public:
    Normal(const json & options);
    double get_aspect_ratio() override {
        return this->aspect_ratio;
    }

    // std::pair<double, double> lonlat_to_xy(double lon, double lat) override;
    std::pair<double, double> xy_to_lonlat(double x, double y) override;

};

}

#endif
