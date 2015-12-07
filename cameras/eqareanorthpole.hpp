
#ifndef VR_LIBMAP_EQAREANORTHPOLE_H_
#define VR_LIBMAP_EQAREANORTHPOLE_H_ value
#ifndef ARCTIC_CIRCLE
	#define ARCTIC_CIRCLE (M_PI / 3)
#endif
#include "./camera.hpp"

namespace vr {

class Eqareanorthpole: public Camera {
public:
    using Camera::Camera;

    double get_aspect_ratio() override {
        return 1.0;
    }

    cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override {
    	if (lonlat.y < ARCTIC_CIRCLE)
	        return cv::Point2d(NAN, NAN);
	    else {
	    	double rho = (M_PI / 2 - lonlat.y) / ARCTIC_CIRCLE;	//rho [0, 0.5]
	    	double x = rho * sin(lonlat.x) + 0.5;
	    	double y = rho * cos(lonlat.x) + 0.5;
	    	return cv::Point2d(x,y);
	    }
    }

    cv::Point2d image_to_obj_single(const cv::Point2d & xy) override {
    	cv::Point2d diff = xy - cv::Point2d(0.5, 0.5);
    	double rho = cv::sqrt(diff.x * diff.x + diff.y * diff.y);
    	double lon = 0;
    	if (rho > 0.5) {
	    	if (xy.x >= 0.5) 
	    		lon = - acos((xy.y - 0.5) / rho);
	    	else
	    		lon = acos((xy.y - 0.5) / rho);
	        return cv::Point2d(lon, ARCTIC_CIRCLE);
	    }
	    else {


	    	if (xy.x >= 0.5) 
	    		lon = - acos((xy.y - 0.5) / rho);
	    	else
	    		lon = acos((xy.y - 0.5) / rho);
	    	double lat = M_PI / 2 - rho * ARCTIC_CIRCLE;

	    	return cv::Point2d(lon, lat);
	    }

    }

};

}

#endif
