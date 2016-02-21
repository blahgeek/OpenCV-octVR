/* 
* @Author: BlahGeek
* @Date:   2016-02-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-22
*/

#ifndef LIVESTITCHING_INPUTS_SELECT_H__
#define LIVESTITCHING_INPUTS_SELECT_H__ value

#include <utility>
#include <vector>
#include <memory>

#include <QCamera>
#include <QCameraViewfinder>
#include <QCameraInfo>
#include <QCameraImageCapture>
#include <QWidget>
#include <QImage>
#include <QGridLayout>
#include <QGroupBox>
#include <QCheckBox>

class InputsSelector: public QObject {
private:
    std::vector<std::unique_ptr<QCamera>> cameras;
    std::vector<std::unique_ptr<QCameraViewfinder>> views;
    QList<QCameraInfo> camera_infos;

    QGridLayout * grid = nullptr;
    std::vector<std::unique_ptr<QCheckBox>> check_boxs;

public:
    InputsSelector(QGridLayout * _grid);

    void start();
    void stop();

    void saveImages();

    std::vector<QCameraInfo> getSelected();

};

#endif
