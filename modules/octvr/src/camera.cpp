/* 
* @Author: BlahGeek
* @Date:   2015-10-20
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-14
*/

#include "./camera.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "./cameras/equirectangular.hpp"
#include "./cameras/normal.hpp"
#include "./cameras/pinhole_cam.hpp"
#include "./cameras/fisheye_cam.hpp"
#include "./cameras/fullframe_fisheye_cam.hpp"
#include "./cameras/stupidoval.hpp"
#include "./cameras/cubic.hpp"
#include "./cameras/eqareanorthpole.hpp"
#include "./cameras/eqareasouthpole.hpp"
#include "./cameras/ocam_fisheye.hpp"

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
    X("ocam_fisheye", OCamFisheyeCamera)
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
        this->rotate_vector[0] =   options["rotation"]["roll"].GetDouble();
        this->rotate_vector[1] = - options["rotation"]["yaw"].GetDouble();
        this->rotate_vector[2] = - options["rotation"]["pitch"].GetDouble();
    }

    cv::Mat rotate_x, rotate_y, rotate_z;
    std::vector<double> v;

    v = rotate_vector; v[1] = v[2] = 0; cv::Rodrigues(v, rotate_x);
    v = rotate_vector; v[0] = v[2] = 0; cv::Rodrigues(v, rotate_y);
    v = rotate_vector; v[0] = v[1] = 0; cv::Rodrigues(v, rotate_z);

    this->rotate_matrix = (rotate_x * rotate_z) * rotate_y;

    if(options.HasMember("rotation_matrix")) {
        for(int h = 0 ; h < 3 ; h += 1)
            for(int w = 0 ; w < 3 ; w += 1)
                this->rotate_matrix.at<double>(h, w) = options["rotation_matrix"][h * 3 + w].GetDouble();
    }

    auto prepare_exclude_mask = [&, this](cv::Scalar initial_val) {
        int width = options["width"].GetInt();
        int height = options["height"].GetInt();
        if(this->exclude_mask.empty()) {
            this->exclude_mask = cv::Mat(height, width, CV_8U);
            this->exclude_mask.setTo(initial_val);
        } else {
            CV_Assert(this->exclude_mask.cols == width);
            CV_Assert(this->exclude_mask.rows == height);
        }
    };

    auto prepare_include_mask = [&, this](cv::Scalar initial_val) {
        int width = options["width"].GetInt();
        int height = options["height"].GetInt();
        if(this->include_mask.empty()) {
            this->include_mask = cv::Mat(height, width, CV_8U);
            this->include_mask.setTo(initial_val);
        } else {
            CV_Assert(this->include_mask.cols == width);
            CV_Assert(this->include_mask.rows == height);
        }
    };

    if(options.HasMember("selection")) {
        prepare_exclude_mask(255);  // exclude all

        int left = options["selection"][0].GetInt();
        int right = options["selection"][1].GetInt();
        int top = options["selection"][2].GetInt();
        int bottom = options["selection"][3].GetInt();

        std::vector<cv::Point2i> points({
            cv::Point2i(left, top),
            cv::Point2i(left, bottom - 1),
            cv::Point2i(right - 1, bottom - 1),
            cv::Point2i(right - 1, top)
        });
        cv::fillPoly(this->exclude_mask, 
                     std::vector<std::vector<cv::Point2i>>({points}), 0);
    }

    if(options.HasMember("exclude_masks")) {
        prepare_exclude_mask(0);  // exclude none
        prepare_include_mask(0);  // include none
        this->draw_mask(options["exclude_masks"], MaskType::exclude);  //PTGui's "exclude_mask" also has "include_mask"
    }

    if(options.HasMember("include_masks")) {
        prepare_include_mask(0);  // include none
        this->draw_mask(options["include_masks"], MaskType::include);  //Hugin's "include_mask"
    }

    if(options.HasMember("longitude_selection")) {
        // max_longitude can be larger than PI
        // to allow ranges like [PI/2, M_PI] + [-PI, -PI/2]
        // which can be [PI/2, PI/2*3]
        this->min_longitude = options["longitude_selection"][0].GetDouble();
        this->max_longitude = options["longitude_selection"][1].GetDouble();
        CV_Assert(this->max_longitude > this->min_longitude);
    } else {
        this->min_longitude = - M_PI;
        this->max_longitude =   M_PI;
    }
}

bool Camera::is_valid_longitude(double longitude) {
    #define BETWEEN(x) ((x) >= this->min_longitude && (x) <= this->max_longitude)
    return BETWEEN(longitude) || BETWEEN(longitude + 2 * M_PI) || BETWEEN(longitude + 4 * M_PI);
    #undef BETWEEN
}

void Camera::draw_mask(const rapidjson::Value & masks, MaskType mask_type) {
    for(auto area = masks.Begin() ; area != masks.End() ; area ++) {
        std::string area_type = (*area)["type"].GetString();
        if(area_type == "polygonal") {
            std::vector<double> args;
            for(auto x = (*area)["args"].Begin() ; x != (*area)["args"].End() ; x ++)
                args.push_back(x->GetDouble());
            std::cerr << "Drawing polygonal mask... " << args.size() / 2 << " points" << std::endl;
            std::vector<cv::Point2i> points;
            for(int i = 0 ; i < args.size() ; i += 2)
                points.emplace_back(int(args[i]), int(args[i+1]));
            switch (mask_type)
            {
                case MaskType::include:
                    cv::fillPoly(this->include_mask, 
                                 std::vector<std::vector<cv::Point2i>>({points}), 255);
                    break;
                case MaskType::exclude:
                    cv::fillPoly(this->exclude_mask, 
                                 std::vector<std::vector<cv::Point2i>>({points}), 255);
                    break;
            }
        }
        else if(area_type == "png") {
            std::cerr << "Drawing PNG image mask... " << std::endl;
            std::vector<unsigned char> args;
            for(auto x = (*area)["args"].Begin() ; x != (*area)["args"].End() ; x ++)
                args.push_back(static_cast<unsigned char>(x->GetInt()));
            cv::Mat mask_img = cv::imdecode(args, 1);
            CV_Assert(mask_img.size() == this->exclude_mask.size());
            CV_Assert(mask_img.type() == CV_8UC3);

            std::vector<cv::Mat> mask_img_channels;
            cv::split(mask_img, mask_img_channels);

            this->exclude_mask.setTo(255, mask_img_channels[2]); // RED channel
            this->include_mask.setTo(255, mask_img_channels[1]); // GREEN channel
        }
        else
            assert(false);
    }
}

cv::Point2d Camera::sphere_xyz_to_lonlat(const cv::Point3d & xyz) {
    auto p = xyz * (1.0 / cv::norm(xyz));
    return cv::Point2d(atan2(-p.z, p.x), asin(p.y));
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
    std::vector<bool> lonlats_valid;
    for(auto & ll: lonlats) {
        xyzs.push_back(sphere_lonlat_to_xyz(ll));
        lonlats_valid.push_back(is_valid_longitude(ll.x));
    }
    // rotate it
    sphere_rotate(xyzs, false);

    // prepare for return value
    std::vector<cv::Point2d> ret;
    ret.reserve(lonlats.size());

    // compute
    for(size_t i = 0 ; i < xyzs.size() ; i += 1) {
        auto ll = sphere_xyz_to_lonlat(xyzs[i]);
        cv::Point2d p = cv::Point2d(NAN, NAN);
        if(lonlats_valid[i])
            p = obj_to_image_single(ll);
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

std::vector<bool> Camera::get_include_mask(const std::vector<cv::Point2d> & lonlats) {
    if (this->include_mask.empty())
        return std::vector<bool>();
    // convert lon/lat to xyz in sphere
    std::vector<cv::Point3d> xyzs;
    xyzs.reserve(lonlats.size());
    for(auto & ll: lonlats)
        xyzs.push_back(sphere_lonlat_to_xyz(ll));
    // rotate it
    sphere_rotate(xyzs, false);

    // prepare for return value
    std::vector<bool> ret;
    ret.reserve(lonlats.size());

    // compute
    for(auto & xyz: xyzs) {
        auto p = obj_to_image_single(sphere_xyz_to_lonlat(xyz));
        bool p_visible = false;
        if(p.x >= 0 && p.x < 1 && p.y >= 0 && p.y < 1) {
            if(!this->exclude_mask.empty()) {
                int W = p.x * this->exclude_mask.cols;
                int H = p.y * this->exclude_mask.rows;
                if(!this->include_mask.empty() && this->include_mask.at<unsigned char>(H, W))
                    p_visible = true;
            }
        }
        ret.push_back(p_visible);
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
