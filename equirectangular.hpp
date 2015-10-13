/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-13
*/

#ifndef VR_LIBMAP_EQUIRECTANGULAR_H
#define VR_LIBMAP_EQUIRECTANGULAR_H value

#include "./libmap.hpp"

namespace vr {

class Equirectangular: public Map {
public:
    using Map::Map;
    
    virtual double get_aspect_ratio() override {
        return 2.0f;
    }
    virtual std::pair<double, double> lonlat_to_xy(double lon, double lat) override;
    virtual std::pair<double, double> xy_to_lonlat(double x, double y) override;
};

}

#endif
