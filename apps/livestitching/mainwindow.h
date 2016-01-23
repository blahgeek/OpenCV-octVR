#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QImage>
#include <QTimer>     // 设置采集数据的间隔时间

#include <highgui.h>  //包含opencv库头文件
#include <cv.h>
#include <QCamera>
#include <QCameraViewfinder>
#include <QCameraImageCapture>

namespace Ui {
class MainWindow;
}

struct caminfo
{
    public:
        int height;
        QString center_dy;
        int width;
        QString center_dx;
        QString exposure_value;
        QString exposure_value_blue;
        QString rotate_1;
        QString rotate_2;
        QString rotate_3;
        QString exposure_value_red;
        QString hfov;
        QString radial_1;
        QString radial_2;
        QString radial_3;
        QString type;
        int device=0;
};

struct overlayinfo
{
public:
    QString aspect_ratio;
    QString cam_opt;
    QString rotate_1;
    QString rotate_2;
    QString rotate_3;
    int device=0;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    caminfo cam[10];
    overlayinfo overlay[10];
    int inputNum;
    int overlaynum;
    explicit MainWindow(QWidget *parent = 0);
    void openPTO();
    void changeCamInfo();
    void overlayinc();
    void overlaydec();
    void changeOverlayInfo();
    void changeasp();
    void changecamopt();
    void changerot1();
    void changerot2();
    void changerot3();
    void generatefile();
    int getint(QString, QString);
    QString getfloat(QString, QString);

    ~MainWindow();

   void openCamera();      // 打开摄像头
   void readFarme();       // 读取当前帧信息
   void bindCamera();     // 关闭摄像头。
   void takingPictures();  // 拍照
   void displayImage(int,QImage);
   void linkcam();
   void generatecommand();


private:
   Ui::MainWindow *ui;
//   QTimer    *timer;
//   QImage    *imag;
//   CvCapture *camera;// 视频获取结构， 用来作为视频获取函数的一个参数
//   IplImage  *frame;//申请IplImage类型指针，就是申请内存空间来存放每一帧图像
   QCamera *camera;
   QCameraViewfinder *viewfinder;
   QCameraImageCapture *imageCapture;

};

#endif // MAINWINDOW_H
