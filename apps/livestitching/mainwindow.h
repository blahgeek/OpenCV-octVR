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
#include <QTemporaryDir>

#include <QTimer>

#include <QMediaPlayer>

#include <vector>
#include <list>
#include <memory>

#include "./qjsonmodel.h"

#include "./inputs_select.hpp"
#include "./pto_template.hpp"
#include "./preview_video.hpp"
#include "./runner.hpp"

namespace Ui {
class MainWindow;
}

#define PREVIEW_WIDTH 1280
#define PREVIEW_HEIGHT 640

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public:
    void run();

public slots:
    void onTabChanged(int index);
    void onInputsSelectChanged();
    void onInputSaveButtonClicked();

    void onTemplateChanged();
    void onRunningStatusChanged();

private:
    Ui::MainWindow *ui;

    std::unique_ptr<InputsSelector> inputs_selector;
    std::unique_ptr<PTOTemplate> pto_template;
    std::unique_ptr<PreviewVideoWidget> preview_video;
    std::unique_ptr<Runner> runner;

    QTimer preview_timer;

    QTemporaryDir temp_dir;

    QVideoWidget * videoWidget;
    QMediaPlayer * videoPreviewer;
};

#endif // MAINWINDOW_H
