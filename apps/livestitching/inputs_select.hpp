/* 
* @Author: BlahGeek
* @Date:   2016-02-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-27
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

    Q_OBJECT

private:
    std::vector<std::unique_ptr<QCamera>> cameras;
    std::vector<std::unique_ptr<QCameraViewfinder>> views;
    QList<QCameraInfo> camera_infos;

    QGridLayout * grid = nullptr;
    std::vector<std::unique_ptr<QCheckBox>> check_boxs;

    int fps = 30;

public slots:
    void onInputsFpsChanged(int _fps);

signals:
    void selectedChanged();

public:
    InputsSelector(QGridLayout * _grid);

    void start();
    void stop();

    void saveImages(int crop_x, int crop_w);

    std::vector<QCameraInfo> getSelected();
    std::vector<QCameraInfo> getAll();

    QStringList getInputArgs();

};

#endif
