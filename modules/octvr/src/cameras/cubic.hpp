/* 
* @Author: BlahGeek
* @Date:   2015-11-14
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-11-14
*/

#ifndef VR_LIBMAP_CUBIC_H_
#define VR_LIBMAP_CUBIC_H_ value

#include "../camera.hpp"

namespace vr {

// See https://code.facebook.com/posts/1638767863078802/under-the-hood-building-360-video/
class Cubic: public Camera {
private:
    // x / y in [-1, 1]
    cv::Point2d cubic_face_to_img(int index, double x, double y) {
        cv::Point2d ret((index % 3) * 1.0 / 3.0, (index / 3) * 1.0 / 2.0);
        ret.x += (x + 1.0) / 2.0 / 3.0;
        ret.y += (y + 1.0) / 2.0 / 2.0;
        return ret;
    }

    std::pair<int, cv::Point2d> img_to_cubic_face(double x, double y) {
        int index_x = 0;
        int index_y = 0;
        if(y >= 0.5) index_y = 1;

        if(x >= 2.0 / 3.0) index_x = 2;
        else if(x >= 1.0 / 3.0) index_x = 1;

        cv::Point2d p((x - index_x * 1.0 / 3.0) * 3.0 * 2.0 - 1.0,
                      (y - index_y * 1.0 / 2.0) * 2.0 * 2.0 - 1.0);
        return std::make_pair(index_y * 3 + index_x, p);
    }

public:
    //using Camera::Camera;
    Cubic(const json & options) :Camera(options){}

    double get_aspect_ratio() override {
        return 3.0 / 2.0;
    }

    cv::Point2d obj_to_image_single(const cv::Point2d & lonlat) override {
        auto point = this->sphere_lonlat_to_xyz(lonlat);
        cv::Point3d scale_point;

        #define WITHIN(P, M, N) (P.M >= -1.0 && P.M <= 1.0 && P.N >= -1.0 && P.N <= 1.0)

        if(fabs(point.x) > 1e-2) {
            scale_point = point / fabs(point.x); // intersect with x = 1 / x = -1
            if(WITHIN(scale_point, y, z)) {
                if(scale_point.x < 0)
                    return cubic_face_to_img(1, -scale_point.z, scale_point.y);
                else 
                    return cubic_face_to_img(0, scale_point.z, scale_point.y);
            }
        }

        if(fabs(point.z) > 1e-2) {
            scale_point = point / fabs(point.z);
            if(WITHIN(scale_point, x, y)) {
                if(scale_point.z < 0)
                    return cubic_face_to_img(4, scale_point.x, scale_point.y);
                else 
                    return cubic_face_to_img(5, -scale_point.x, scale_point.y);
            }
        }

        if(fabs(point.y) > 1e-2) {
            scale_point = point / fabs(point.y);
            if(WITHIN(scale_point, x, z)) {
                if(scale_point.y < 0)
                    return cubic_face_to_img(2, scale_point.x, - scale_point.z);
                else 
                    return cubic_face_to_img(3, scale_point.x, scale_point.z);
            }
        }

        return cv::Point2d(NAN, NAN);
    }

    cv::Point2d image_to_obj_single(const cv::Point2d & xy) override {
        auto cubic_face = this->img_to_cubic_face(xy.x, xy.y);
        int face_index = cubic_face.first;
        cv::Point2d face_p = cubic_face.second;

        switch(face_index) {
            case 0: return sphere_xyz_to_lonlat(cv::Point3d(1.0, face_p.y, face_p.x));
            case 1: return sphere_xyz_to_lonlat(cv::Point3d(-1., face_p.y, -face_p.x));
            case 2: return sphere_xyz_to_lonlat(cv::Point3d(face_p.x, -1., -face_p.y));
            case 3: return sphere_xyz_to_lonlat(cv::Point3d(face_p.x, 1.0, face_p.y));
            case 4: return sphere_xyz_to_lonlat(cv::Point3d(face_p.x, face_p.y, -1.0));
            case 5: return sphere_xyz_to_lonlat(cv::Point3d(-face_p.x, face_p.y, 1.0));
            default: assert(false);
        }

        return cv::Point2d(NAN, NAN);

    }

};

}

#endif
