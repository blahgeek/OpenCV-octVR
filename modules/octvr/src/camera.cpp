/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-21
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "./camera.hpp"
#include "./cameras/equirectangular.hpp"
#include "./cameras/normal.hpp"
#include "./cameras/pinhole_cam.hpp"
#include "./cameras/fisheye_cam.hpp"
#include "./cameras/fullframe_fisheye_cam.hpp"
#include "./cameras/stupidoval.hpp"
#include "./cameras/cubic.hpp"
#include "./cameras/eqareanorthpole.hpp"
#include "./cameras/eqareasouthpole.hpp"

using namespace vr;

std::unique_ptr<Camera> Camera::New(const std::string & type, const rapidjson::Value & options) {
    #define X(s, t) \
        else if (type == s) return std::unique_ptr<Camera>(new t(options));

    if(false){}

    X("normal", Normal)
    X("pinhole", PinholeCamera)
    X("fisheye", FisheyeCamera)
    X("equirectangular", Equirectangular)
    X("fullframe_fisheye", FullFrameFisheyeCamera)
    X("stupidoval", StupidOval)
    X("cubic", Cubic)
    X("eqareanorthpole", Eqareanorthpole)
    X("eqareasouthpole", Eqareasouthpole)
    return nullptr;

    #undef X
}

Camera::Camera(const rapidjson::Value & options) {
    this->rotate_vector = std::vector<double>({0, 0, 0});
    if(options.HasMember("rotation")) {
        this->rotate_vector[0] = options["rotation"]["roll"].GetDouble();
        this->rotate_vector[1] = options["rotation"]["yaw"].GetDouble();
        this->rotate_vector[2] = options["rotation"]["pitch"].GetDouble();
    }

    cv::Mat rotate_x, rotate_y, rotate_z;
    std::vector<double> v;

    v = rotate_vector; v[1] = v[2] = 0; cv::Rodrigues(v, rotate_x);
    v = rotate_vector; v[0] = v[2] = 0; cv::Rodrigues(v, rotate_y);
    v = rotate_vector; v[0] = v[1] = 0; cv::Rodrigues(v, rotate_z);

    this->rotate_matrix = (rotate_x * rotate_z) * rotate_y;

    if(options.HasMember("exclude_masks")) {
        int width = options["width"].GetInt();
        int height = options["height"].GetInt();
        this->exclude_mask = cv::Mat(height, width, CV_8U);
        this->exclude_mask.setTo(0);
        this->drawExcludeMask(options["exclude_masks"]);
    }
}

void Camera::drawExcludeMask(const rapidjson::Value & masks) {
    for(auto area = masks.Begin() ; area != masks.End() ; area ++) {
        std::string area_type = (*area)["type"].GetString();
        std::vector<double> args;
        for(auto x = (*area)["args"].Begin() ; x != (*area)["args"].End() ; x ++)
            args.push_back(x->GetDouble());
        if(area_type == "polygonal") {
            std::cerr << "Drawing polygonal mask... " << args.size() / 2 << " points" << std::endl;
            std::vector<cv::Point2i> points;
            for(int i = 0 ; i < args.size() ; i += 2)
                points.emplace_back(int(args[i]), int(args[i+1]));
            cv::fillPoly(this->exclude_mask, 
                         std::vector<std::vector<cv::Point2i>>({points}), 255);
        }
        else
            assert(false);
    }
}

cv::Point2d Camera::sphere_xyz_to_lonlat(const cv::Point3d & xyz) {
    auto p = xyz * (1.0 / cv::norm(xyz));
    return cv::Point2d(-atan2(p.z, p.x), asin(p.y));
}

cv::Point3d Camera::sphere_lonlat_to_xyz(const cv::Point2d & lonlat) {
    auto lon = lonlat.x;
    auto lat = lonlat.y;
    return cv::Point3d(cos(lon) * cos(lat),
                       sin(lat),
                       -sin(lon) * cos(lat));
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
    for(auto & xyz: xyzs) {
        auto p = obj_to_image_single(sphere_xyz_to_lonlat(xyz));
        if(p.x >= 0 && p.x < 1 && p.y >= 0 && p.y < 1) {
            if(!this->exclude_mask.empty()) {
                int W = p.x * this->exclude_mask.cols;
                int H = p.y * this->exclude_mask.rows;
                if(this->exclude_mask.at<unsigned char>(H, W))
                    p = cv::Point2d(NAN, NAN);
            }
        }
        ret.push_back(p);
    }

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
