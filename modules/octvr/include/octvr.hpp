/* 
* @Author: BlahGeek
* @Date:   2015-10-13
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-05-05
*/

#ifndef VR_LIBMAP_BASE_H
#define VR_LIBMAP_BASE_H value

#include "rapidjson/document.h"
#include <utility>
#include <cmath>
#include <cassert>
#include <exception>
#include <vector>
#include <queue>
#include <string>
#include <tuple>
#include <fstream>
#include <stdint.h>
#include <iostream>
#include <mutex>
#include <condition_variable>
#if defined(__linux__)|| defined(__APPLE__)
#include <sys/time.h>
#elif defined(ANDROID)
#include <time.h>
#elif defined(_WIN32)
#include <windows.h>
#include <time.h>
#endif

#include "opencv2/core.hpp"


namespace vr {

class CV_EXPORTS_W CameraInterface {
public:
    virtual std::vector<cv::Point2d> obj_to_image(const std::vector<cv::Point2d> & lonlats) = 0;
    virtual std::vector<cv::Point2d> image_to_obj(const std::vector<cv::Point2d> & xys) = 0;
    virtual ~CameraInterface() {}
};

// Multiple input -> single output
class CV_EXPORTS_W MapperTemplate {
public:
    std::string out_type;
    const rapidjson::Value * out_opts = nullptr;
    // const rapidjson::Value & out_opts;
    cv::Size out_size;

    typedef struct {
        cv::Rect roi; // related to out_size
        cv::Mat map1, map2;
        cv::Mat mask;
        cv::Mat vignette;
    } Input;

    std::vector<Input> inputs;
    std::vector<Input> overlay_inputs;

    std::vector<cv::Mat> seam_masks;  // only for inputs (not overlay_inputs)
    std::vector<bool> visible_mask; // only used in dumper (for green mask of PTGui)

    CameraInterface * output_cam = nullptr;
    std::vector<CameraInterface *> input_cams;

public:
    // Create new template
    // width/height must be suitable to output model
    MapperTemplate(const std::string & to,
                   const rapidjson::Value & to_opts,
                   int width, int height);
    void add_input(const std::string & from, 
                   const rapidjson::Value & from_opts, 
                   bool overlay=false,
                   bool use_roi=true);
    // Prepare seam masks with provided images (optional)
    void create_masks(const std::vector<cv::Mat> & imgs = std::vector<cv::Mat>());
    void morph_controlpoints(const rapidjson::Value & control_points);
    
    void dump(std::ofstream & f);

    // Load existing template
    explicit MapperTemplate(std::ifstream & f);
    ~MapperTemplate();
};

#define OCTVR_PREVIEW_DATA0_MEMORY_KEY "opencv_octvr_preview_0"
#define OCTVR_PREVIEW_DATA1_MEMORY_KEY "opencv_octvr_preview_1"
#define OCTVR_PREVIEW_DATA_META_MEMORY_KEY "opencv_octvr_preview_meta"

struct CV_EXPORTS_W PreviewDataHeader {
    int width, height;
    int step;
    double fps;
};

#define OCTVR_UYVY422 1
#define OCTVR_YUYV422 2

class CV_EXPORTS_W AsyncMultiMapper {
public:
    static AsyncMultiMapper * New(const std::vector<MapperTemplate> & mts,
                                  std::vector<cv::Size> in_sizes,
                                  cv::Size out_size,
                                  std::vector<int> blend_modes,
                                  std::vector<int> gain_modes,
                                  std::vector<cv::Rect_<double>> output_regions,
                                  int input_pix_fmt,
                                  cv::Size preview_size);

    /**
     * Push one frame
     * @param inputs Input images
     * @param output Output images, in UYVY422
     */
    virtual void push(std::vector<cv::Mat> & inputs,
                      cv::Mat & output) = 0;
    virtual void pop() = 0;

    virtual ~AsyncMultiMapper() {}
};

class CV_EXPORTS_W FastMapper {

private:
    std::vector<cv::Size> in_sizes;
    cv::Size out_size;

    std::vector<cv::UMat> map1s, half_map1s;
    std::vector<cv::UMat> map2s, half_map2s;
    std::vector<cv::UMat> masks, half_masks;
    std::vector<cv::UMat> feather_masks, half_feather_masks;

private:
    cv::UMat output_s_c0;
    std::vector<cv::UMat> output_s_c1c2;
    std::vector<cv::UMat> input_c1c2;
    cv::UMat output_c1c2_merge;

public:
    FastMapper(const MapperTemplate & mt, std::vector<cv::Size> in_sizes);
    void stitch(const std::vector<cv::UMat> & inputs, cv::UMat & output);
    void stitch_nv12(const std::vector<cv::UMat> & inputs, cv::UMat & output);
};

class CV_EXPORTS_W Timer {
protected:
    int64_t t;
    std::string name;

#if defined(_WIN32)
	int64_t frequency;
#endif

public:
    explicit Timer(std::string name);
    Timer();
    double tick(std::string msg);
};

template <class T>
class CV_EXPORTS_W Queue {
private:
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cond_empty;
public:
    bool empty() { return q.empty() ;}
    void push(T&& val) {
        std::lock_guard<std::mutex> guard(mtx);
        q.push(std::forward<T>(val));
        cond_empty.notify_one();
    }
    void push(const T & val) { this->push(T(val)); }
    T pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cond_empty.wait(lock, [this](){ return !this->empty(); });
        T ret = q.front();
        q.pop();
        return ret;
    }
};

}

#endif
