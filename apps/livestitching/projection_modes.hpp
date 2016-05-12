/* 
* @Author: BlahGeek
* @Date:   2016-04-27
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-27
*/

#ifndef VR_LIVESTITCHING_PROJECTION_MODES_H__
#define VR_LIVESTITCHING_PROJECTION_MODES_H__ value

#include <vector>
#include <tuple>
#include <QJsonDocument>
#include <QString>
#include <QStringList>

#include "octvr.hpp"

struct _ProjectionModeOutput {
    cv::Rect_<double> region;
    bool should_use_multiband;
    bool should_compute_exposure;

    int input_id; // left(0) or right(1)
    QJsonDocument output_options;
};

using ProjectionMode = std::vector<struct _ProjectionModeOutput>;

extern ProjectionMode PROJECTION_MODE_MONO360;
extern ProjectionMode PROJECTION_MODE_3DV;
extern ProjectionMode PROJECTION_MODE_3DV_CYLINDER_SLICE_2X25_3DV;

#endif
