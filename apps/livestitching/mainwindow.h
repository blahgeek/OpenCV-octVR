#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QImage>
#include <QTimer>     // 设置采集数据的间隔时间

#include <QCamera>
#include <QCameraViewfinder>
#include <QCameraImageCapture>

#include <vector>
#include <list>
#include <memory>

#include "./qjsonmodel.h"

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
    void gotoStitch();

    void reEditPTO();

    void run();

public slots:
    void removeImageCapture(int id, const QString & fileName);

private:
    Ui::MainWindow *ui;

    QJsonModel json_model;
    QList<QCameraInfo> camera_infos;

    using CameraAndView = std::tuple<std::unique_ptr<QCamera>, std::unique_ptr<QCameraViewfinder>, QString>;
    std::vector<CameraAndView> input_cameras, overlay_cameras;
    std::map<int, QCameraImageCapture *> image_captures;

    QProcess ffmpeg_proc;
};

#endif // MAINWINDOW_H
