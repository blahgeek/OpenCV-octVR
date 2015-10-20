/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2015-10-20
*/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "./camera.hpp"
#include "./equirectangular.hpp"
#include "./normal.hpp"
#include "./pinhole_cam.hpp"
#include "./fisheye_cam.hpp"

using namespace vr;

std::unique_ptr<Camera> Camera::New(const std::string & type, const json & options) {
    #define X(s, t) \
        else if (type == s) return std::unique_ptr<Camera>(new t(options));

    if(false){}

    X("normal", Normal)
    X("pinhole", PinholeCamera)
    X("fisheye", FisheyeCamera)
    X("equirectangular", Equirectangular)

    return nullptr;

    #undef X
}

Camera::Camera(const json & options) {
    this->rotate_vector = std::vector<double>({0, 0, 0});
    if(options.find("rotate") != options.end())
        this->rotate_vector = options["rotate"].get<std::vector<double>>();
    cv::Rodrigues(rotate_vector, this->rotate_matrix);
}

cv::Point2d Camera::sphere_xyz_to_lonlat(const cv::Point3d & xyz) {
    auto p = xyz * (1.0 / cv::norm(xyz));
    return cv::Point2d(atan2(p.z, p.x), asin(p.y));
}

cv::Point3d Camera::sphere_lonlat_to_xyz(const cv::Point2d & lonlat) {
    auto lon = lonlat.x;
    auto lat = lonlat.y;
    return cv::Point3d(cos(lon) * cos(lat),
                       sin(lat),
                       sin(lon) * cos(lat));
}

void Camera::sphere_rotate(std::vector<cv::Point3d> & points, bool reverse) {
    cv::Mat m(points.size(), 3, CV_64F, points.data(), sizeof(cv::Point3d));
    cv::Mat r = rotate_matrix;
    if(reverse) r = r.inv();

    cv::Mat rotated = m * r.t();
    rotated.copyTo(m);
    assert(m.data == static_cast<void *>(points.data()));
}

std::vector<cv::Point2d> Camera::obj_to_image(const std::vector<cv::Point2d> & lonlats) {
    // convert lon/lat to xyz in sphere
    std::vector<cv::Point3d> xyzs;
    xyzs.reserve(lonlats.size());
    for(auto & ll: lonlats)
        xyzs.push_back(sphere_lonlat_to_xyz(ll));
    // rotate it
    sphere_rotate(xyzs, false);

    // prepare for return value
    std::vector<cv::Point2d> ret;
    ret.reserve(lonlats.size());

    // compute
    for(auto & xyz: xyzs)
        ret.push_back(obj_to_image_single(sphere_xyz_to_lonlat(xyz)));
    return ret;
}

std::vector<cv::Point2d> Camera::image_to_obj(const std::vector<cv::Point2d> & xys) {
    std::vector<cv::Point3d> points;
    points.reserve(xys.size());
    for(auto & xy: xys)
        points.push_back(sphere_lonlat_to_xyz(image_to_obj_single(xy)));

    // rotate it
    sphere_rotate(points, true);

    // convert back
    std::vector<cv::Point2d> ret;
    ret.reserve(points.size());
    for(auto & p: points)
        ret.push_back(sphere_xyz_to_lonlat(p));
    return ret;
}
