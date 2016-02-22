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

#include <QMediaPlayer>

#include <vector>
#include <list>
#include <memory>

#include "./qjsonmodel.h"

#include "./inputs_select.hpp"
#include "./pto_template.hpp"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public:
    void run();

    void initPreview();
    void startPreview();
    void stopPreview();

public slots:
    void onTabChanged(int index);
    void onInputsSelectChanged();
    void onInputSaveButtonClicked();

    void onTemplateChanged();

private:
    Ui::MainWindow *ui;

    std::unique_ptr<InputsSelector> inputs_selector;
    std::unique_ptr<PTOTemplate> pto_template;

    QTemporaryDir temp_dir;

    QProcess ffmpeg_proc;
    QString temp_path;
    QString hugin_path;

    QVideoWidget * videoWidget;
    QMediaPlayer * videoPreviewer;
};

#endif // MAINWINDOW_H
