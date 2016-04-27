/* 
* @Author: BlahGeek
* @Date:   2016-04-27
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-27
*/

#include <iostream>
#include "./projection_modes.hpp"
#include <QByteArray>

ProjectionMode PROJECTION_MODE_MONO360 = {
    {cv::Rect_<double>(0., 0., 1., 1.), true, true, 0, 
        QJsonDocument::fromJson(QByteArray(R"JSON(
                                            {
                                                "type": "equirectangular",
                                                "options": {}
                                            }
                                            )JSON")),
    },
};

ProjectionMode PROJECTION_MODE_3DV = {
    {cv::Rect_<double>(0., 0., 1., 0.5), true, true, 0, 
        QJsonDocument::fromJson(QByteArray(R"JSON(
                                            {
                                                "type": "equirectangular",
                                                "options": {}
                                            }
                                            )JSON")),
    },
    {cv::Rect_<double>(0., 0.5, 1., 0.5), true, true, 1,
        QJsonDocument::fromJson(QByteArray(R"JSON(
                                            {
                                                "type": "equirectangular",
                                                "options": {}
                                            }
                                            )JSON")),
    },
};

// 2304x1024
ProjectionMode PROJECTION_MODE_3DV_CYLINDER_SLICE_2X25_3DV = {
    {cv::Rect_<double>(0., 0., 2048./2304., 512./1024.), true, true, 0, 
        QJsonDocument::fromJson(QByteArray(R"JSON(
                                            {
                                                "type": "equirectangular",
                                                "options": {
                                                    "max_lat": 0.78539815,
                                                    "min_lat": -0.78539815
                                                }
                                            }
                                            )JSON")),
    },
    {cv::Rect_<double>(2048./2304., 0., 256./2304., 256./1024.), false, false, 0,
        QJsonDocument::fromJson(QByteArray(R"JSON(
                                            {
                                                "options": {
                                                    "arctic_circle": 0.785398
                                                },
                                                "type": "eqareanorthpole"
                                            }
                                            )JSON")),
    },
    {cv::Rect_<double>(2048./2304., 256./2304., 256./2304., 256./1024.), false, false, 0,
        QJsonDocument::fromJson(QByteArray(R"JSON(
                                            {
                                                "options": {
                                                    "antarctic_circle": -0.785398
                                                },
                                                "type": "eqareasouthpole"
                                            }
                                            )JSON")),
    },
    {cv::Rect_<double>(0., 512./1024., 2048./2304., 512./1024.), true, true, 1,
        QJsonDocument::fromJson(QByteArray(R"JSON(
                                            {
                                                "type": "equirectangular",
                                                "options": {
                                                    "max_lat": 0.78539815,
                                                    "min_lat": -0.78539815
                                                }
                                            }
                                            )JSON")),
    },
    {cv::Rect_<double>(2048./2304., 512./1024., 256./2304., 256./1024.), false, false, 1,
        QJsonDocument::fromJson(QByteArray(R"JSON(
                                            {
                                                "options": {
                                                    "arctic_circle": 0.785398
                                                },
                                                "type": "eqareanorthpole"
                                            }
                                            )JSON")),
    },
    {cv::Rect_<double>(2048./2304., 768./2304., 256./2304., 256./1024.), false, false, 1,
        QJsonDocument::fromJson(QByteArray(R"JSON(
                                            {
                                                "options": {
                                                    "antarctic_circle": -0.785398
                                                },
                                                "type": "eqareasouthpole"
                                            }
                                            )JSON")),
    },
};
