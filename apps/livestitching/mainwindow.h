#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QImage>
#include <QTimer>     // 设置采集数据的间隔时间

#include <QCamera>
#include <QCameraViewfinder>
#include <QCameraImageCapture>
#include <QBoxLayout>

#include <QMediaPlayer>

#include <vector>
#include <list>
#include <memory>

#include "./qjsonmodel.h"

#include "./inputs_select.hpp"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    void openPTO();

    ~MainWindow();

public:
    void jsonAddOverlay();
    void jsonDelOverlay();

    void deviceAddCamera();
    void deviceDelCamera();

    void loadPTO(const QString & filename);
    void locateHugin(); 
    void gotoStitch();

    void reEditPTO();
    void saveAsPTO();

    void run();

    void initPreview();
    void startPreview();
    void stopPreview();

    void onTabChanged(int index);
    void onInputsSelectChanged();

private:
    Ui::MainWindow *ui;

    std::unique_ptr<InputsSelector> inputs_selector;

    QHBoxLayout * inputs_layout = nullptr;

    QJsonModel json_model;
    QList<QCameraInfo> camera_infos;

    using CameraAndView = std::tuple<std::unique_ptr<QCamera>, std::unique_ptr<QCameraViewfinder>, QString>;
    std::vector<CameraAndView> input_cameras, overlay_cameras;
    std::map<int, QCameraImageCapture *> image_captures;

    QProcess ffmpeg_proc;
    QString temp_path;
    QString hugin_path;

    QVideoWidget * videoWidget;
    QMediaPlayer * videoPreviewer;
};

#endif // MAINWINDOW_H
