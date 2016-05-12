/* 
* @Author: BlahGeek
* @Date:   2016-05-04
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-05-07
*/

#include <iostream>
#include "octvr.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"

using namespace vr;

struct ControlPoint {
    int n0, n1;
    cv::Point2f src0, src1;
    cv::Point2f dst0, dst1;
    cv::Point2f dst_mid;
};

static std::vector<cv::Vec6f> getTriangleList(std::vector<cv::Point2f> vertexes) {
    cv::Subdiv2D subdiv(cv::Rect(0,0,1,1));
    subdiv.insert(vertexes);
    std::vector<cv::Vec6f> result, filtered;
    subdiv.getTriangleList(result);
    for(auto x: result) {
        if(!std::all_of(&x[0], &x[6], [](float n){return n >= 0.0 && n <= 1.0;}))
            continue;
        // if(std::fabs(x[0] - x[2]) > 0.2 ||
        //    std::fabs(x[2] - x[4]) > 0.2 ||
        //    std::fabs(x[0] - x[4]) > 0.2)
        //     continue;
        // if(std::fabs(x[1] - x[3]) > 0.7 ||
        //    std::fabs(x[3] - x[5]) > 0.7 ||
        //    std::fabs(x[1] - x[5]) > 0.7)
        //     continue;
        filtered.push_back(x);
    }
    return filtered;
}

static std::vector<cv::Vec3i> getTriangleListIndexes(const std::vector<cv::Vec6f> & triangleList, 
                                              const std::vector<cv::Point2f> & vertexes) {
    std::vector<cv::Vec3i> ret;
    for(auto tri: triangleList) {
        auto cmp = [](cv::Point2f a, cv::Point2f b) {
            return a == b;
            // return std::fabs(a.x - b.x) + std::fabs(a.y - b.y) < 1e-5;
        };
        int i0 = std::find_if(vertexes.begin(), vertexes.end(), [&](cv::Point2f x) {return cmp(x, cv::Point2f(tri[0], tri[1])); }) - vertexes.begin();
        int i1 = std::find_if(vertexes.begin(), vertexes.end(), [&](cv::Point2f x) {return cmp(x, cv::Point2f(tri[2], tri[3])); }) - vertexes.begin();
        int i2 = std::find_if(vertexes.begin(), vertexes.end(), [&](cv::Point2f x) {return cmp(x, cv::Point2f(tri[4], tri[5])); }) - vertexes.begin();
        ret.emplace_back(i0, i1, i2);
    }
    return ret;
}

static std::vector<cv::Vec6f> getTriangleListFromIndexes(const std::vector<cv::Point2f> & vertexes,
                                                  const std::vector<cv::Vec3i> & indexes) {
    std::vector<cv::Vec6f> ret;
    for(auto tri: indexes)
        ret.emplace_back(vertexes[tri[0]].x, vertexes[tri[0]].y,
                         vertexes[tri[1]].x, vertexes[tri[1]].y,
                         vertexes[tri[2]].x, vertexes[tri[2]].y);
    return ret;
}

void MapperTemplate::morph_controlpoints(const rapidjson::Value & control_points) {
    std::vector<cv::UMat> umasks(inputs.size());
    std::vector<cv::Point> corners;
    for(size_t i = 0 ; i < inputs.size() ; i += 1) {
        inputs[i].mask.copyTo(umasks[i]);
        corners.push_back(inputs[i].roi.tl());
    }
    auto seam_finders = new cv::detail::DistanceSeamFinder(2);
    seam_finders->find(std::vector<cv::UMat>(), corners, umasks);
    auto distances = seam_finders->getDistances();
    delete seam_finders;

    std::vector<cv::Mat> masks_h;
    for(auto m: umasks)
        masks_h.push_back(m.getMat(cv::ACCESS_RW));
    // prepare masks done

    auto _translate = [&](float x, float y, int n) {
        auto ll = input_cams[n]->image_to_obj(std::vector<cv::Point2d>({cv::Point2d(x, y)}));
        auto xy = output_cam->obj_to_image(ll);
        return cv::Point2f(xy[0].x, xy[0].y);
    };

    std::vector<struct ControlPoint> cps;
    for(auto x = control_points.Begin() ; x != control_points.End() ; x ++ ) {
        struct ControlPoint cp;
        auto a = x->GetArray();
        cp.n0 = a[0].GetInt();
        cp.n1 = a[1].GetInt();
        cp.src0.x = a[2].GetDouble();
        cp.src0.y = a[3].GetDouble();
        cp.src1.x = a[4].GetDouble();
        cp.src1.y = a[5].GetDouble();
        CV_Assert(cp.n0 < cp.n1);

        cp.dst0 = _translate(cp.src0.x, cp.src0.y, cp.n0);
        cp.dst1 = _translate(cp.src1.x, cp.src1.y, cp.n1);

        int dst0_local_x = cp.dst0.x * out_size.width - inputs[cp.n0].roi.x;
        int dst0_local_y = cp.dst0.y * out_size.height - inputs[cp.n0].roi.y;
        int dst1_local_x = cp.dst1.x * out_size.width - inputs[cp.n1].roi.x;
        int dst1_local_y = cp.dst1.y * out_size.height - inputs[cp.n1].roi.y;

        // if(dst0_local_x < 0 || dst0_local_x >= inputs[cp.n0].roi.width ||
        //    dst0_local_y < 0 || dst0_local_y >= inputs[cp.n0].roi.height)
        //     continue;
        // if(dst1_local_x < 0 || dst1_local_x >= inputs[cp.n1].roi.width ||
        //    dst1_local_y < 0 || dst1_local_y >= inputs[cp.n1].roi.height)
        //     continue;

        // if(masks_h[cp.n0].at<unsigned char>(dst0_local_y, dst0_local_x) == 0 ||
        //    masks_h[cp.n1].at<unsigned char>(dst1_local_y, dst1_local_x) == 0)
        //     continue;

        if(std::fabs(cp.dst0.x - cp.dst1.x) + std::fabs(cp.dst0.y - cp.dst1.y) > 0.1)
            continue;

        float weight0 = distances[cp.n0].at<float>(dst0_local_y, dst0_local_x);
        float weight1 = distances[cp.n1].at<float>(dst1_local_y, dst1_local_x);

        if(weight0 + weight1 < 1e-3)
            weight0 = weight1 = 1.0;

        cp.dst_mid.x = (cp.dst0.x * weight0 + cp.dst1.x * weight1) / (weight0 + weight1);
        cp.dst_mid.y = (cp.dst0.y * weight0 + cp.dst1.y * weight1) / (weight0 + weight1);

        cps.push_back(cp);
    }
    std::cerr << "Number of control points: " << cps.size() << std::endl;


    for(size_t i = 0 ; i < inputs.size() ; i += 1) {
        std::vector<cv::Point2f> src_vertexes, dst_vertexes;
        for(auto cp: cps) {
            if(cp.n0 == i) {
                src_vertexes.push_back(cp.dst0);
                dst_vertexes.push_back(cp.dst_mid);
            }
            if(cp.n1 == i) {
                src_vertexes.push_back(cp.dst1);
                dst_vertexes.push_back(cp.dst_mid);
            }
        }

        float bb_left=1., bb_right=0., bb_top=1., bb_bottom=0.;
        for(auto v: src_vertexes) {
            bb_left = std::min(bb_left, v.x);
            bb_right = std::max(bb_right, v.x);
            bb_top = std::min(bb_top, v.y);
            bb_bottom = std::max(bb_bottom, v.y);
        }
        for(auto v: dst_vertexes) {
            bb_left = std::min(bb_left, v.x);
            bb_right = std::max(bb_right, v.x);
            bb_top = std::min(bb_top, v.y);
            bb_bottom = std::max(bb_bottom, v.y);
        }
        bb_left = std::max(1e-3, bb_left - 0.05);
        bb_top = std::max(1e-3, bb_top - 0.05);
        bb_right = std::min(1 - 1e-3, bb_right + 0.05);
        bb_bottom = std::min(1 - 1e-3, bb_bottom + 0.05);

        for(float x = bb_left ; x < bb_right + 1e-3 ; x += (bb_right - bb_left) / 10) {
            src_vertexes.emplace_back(x, bb_top);
            src_vertexes.emplace_back(x, bb_bottom);
            dst_vertexes.emplace_back(x, bb_top);
            dst_vertexes.emplace_back(x, bb_bottom);
        }
        for(float y = bb_top + (bb_bottom - bb_top) / 10; y < bb_bottom - (bb_bottom - bb_top) / 10 + 1e-3 ; y += (bb_bottom - bb_top) / 10) {
            src_vertexes.emplace_back(bb_left, y);
            src_vertexes.emplace_back(bb_right, y);
            dst_vertexes.emplace_back(bb_left, y);
            dst_vertexes.emplace_back(bb_right, y);
        }

        // auto B = [&](float x, float y) { return cv::Point2f((x * umasks[i].cols + inputs[i].roi.x) / out_size.width,
        //                                                     (y * umasks[i].rows + inputs[i].roi.y) / out_size.height); };
        // for(float x = 1e-3 ; x < 1.0 ; x += 0.19979) {
        //     src_vertexes.push_back(B(x, 1e-3));
        //     dst_vertexes.push_back(B(x, 1e-3));
        //     src_vertexes.push_back(B(x, 1.0 - 1e-3));
        //     dst_vertexes.push_back(B(x, 1.0 - 1e-3));
        // }
        // for(float y = 0.2 ; y < 0.9 ; y += 0.2) {
        //     src_vertexes.push_back(B(1e-3, y));
        //     dst_vertexes.push_back(B(1e-3, y));
        //     src_vertexes.push_back(B(1.0 - 1e-3, y));
        //     dst_vertexes.push_back(B(1.0 - 1e-3, y));
        // }
        inputs[i].src_triangles = getTriangleList(src_vertexes);
        inputs[i].dst_triangles = getTriangleListFromIndexes(dst_vertexes, 
                                        getTriangleListIndexes(inputs[i].src_triangles, src_vertexes));

        auto T = [&](float x, float y) { return cv::Point2f(x * out_size.width - inputs[i].roi.x,
                                                            y * out_size.height - inputs[i].roi.y);};
        cv::Mat new_map1 = inputs[i].map1.clone();
        cv::Mat new_map2 = inputs[i].map2.clone();
        cv::Mat new_mask = inputs[i].mask.clone();
        for(int k = 0 ; k < inputs[i].src_triangles.size() ; k += 1) {
            auto src = inputs[i].src_triangles[k];
            auto dst = inputs[i].dst_triangles[k];
            cv::Point2f src_ts[] = {T(src[0], src[1]), T(src[2], src[3]), T(src[4], src[5])};
            cv::Point2f dst_ts[] = {T(dst[0], dst[1]), T(dst[2], dst[3]), T(dst[4], dst[5])};
            auto warp_mat = cv::getAffineTransform(src_ts, dst_ts);

            cv::Mat tri_mask(inputs[i].mask.size(), CV_8U);
            tri_mask.setTo(0);
            std::vector<cv::Point> round_dst_ts;
            round_dst_ts.emplace_back(std::round(dst_ts[0].x), std::round(dst_ts[0].y));
            round_dst_ts.emplace_back(std::round(dst_ts[1].x), std::round(dst_ts[1].y));
            round_dst_ts.emplace_back(std::round(dst_ts[2].x), std::round(dst_ts[2].y));
            cv::fillPoly(tri_mask, std::vector<std::vector<cv::Point>>({round_dst_ts}), 255);

            cv::Mat tmp_map(inputs[i].mask.size(), inputs[i].map1.type());
            cv::warpAffine(inputs[i].map1, tmp_map, warp_mat, inputs[i].mask.size());
            tmp_map.copyTo(new_map1, tri_mask);
            cv::warpAffine(inputs[i].map2, tmp_map, warp_mat, inputs[i].mask.size());
            tmp_map.copyTo(new_map2, tri_mask);

            cv::Mat tmp_mask(inputs[i].mask.size(), inputs[i].mask.type());
            cv::warpAffine(inputs[i].mask, tmp_map, warp_mat, inputs[i].mask.size());
            tmp_map.copyTo(new_mask, tri_mask);
        }
        inputs[i].map1 = new_map1;
        inputs[i].map2 = new_map2;
        inputs[i].mask = new_mask;
    }

}
