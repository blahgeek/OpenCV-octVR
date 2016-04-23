/* 
* @Author: BlahGeek
* @Date:   2016-02-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-23
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
#include <QAudioDeviceInfo>
#include <QWidget>
#include <QImage>
#include <QGridLayout>
#include <QGroupBox>
#include <QCheckBox>
#include <QComboBox>

class InputsSelector: public QObject {

    Q_OBJECT

private:
    std::vector<std::unique_ptr<QCamera>> cameras;
    std::vector<std::unique_ptr<QCameraViewfinder>> views;
    QList<QCameraInfo> camera_infos;
    QList<QAudioDeviceInfo> audio_infos;

    QGridLayout * grid = nullptr;
    QComboBox * audio_combo = nullptr;
    std::vector<std::unique_ptr<QCheckBox>> check_boxs;

    int fps = 30;

public slots:
    void onInputsFpsChanged(int _fps);

signals:
    void selectedChanged();

public:
    InputsSelector(QGridLayout * _grid, QComboBox * _audio_combo);

    void start();
    void stop();

    void saveImages(int crop_x, int crop_w);

    std::vector<QCameraInfo> getSelected();
    std::vector<QCameraInfo> getAll();

    QStringList getInputArgs();

};

#endif
