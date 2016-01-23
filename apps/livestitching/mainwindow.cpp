
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include "qfile.h"
#include <QTextStream>
#include <QProcess>
#include <QDebug>
#include <QRegExp>
#include <QLineEdit>
#include <QCameraInfo>

int MainWindow::getint(QString inputline,QString cap)
{
    QRegExp rx1("(-?\\d+)(\\.\\d+)");
    QRegExp rx2("\\d+");

    int heightstart=inputline.indexOf(QRegExp("\\b"+cap+"[\\w=]*\\b"));
    int heightend=inputline.indexOf(" ",heightstart);
    QString heightnum=inputline.mid(heightstart,heightend-heightstart);
    int pos=heightnum.indexOf(rx2);

    return rx2.cap(0).toInt();
}

QString MainWindow::getfloat(QString inputline,QString cap)
{
    QRegExp rx1("-?[\\d\\.]+");
    QRegExp rx2("\\d+");

    int heightstart=inputline.indexOf(QRegExp("\\b"+cap+"[\\w=]*\\b"));
    int heightend=inputline.indexOf(" ",heightstart);
    QString heightnum=inputline.mid(heightstart,heightend-heightstart);
    int pos=heightnum.indexOf(rx1);

    return rx1.cap(0);
}

void MainWindow::openPTO()
{
    //QFile datafile;
    //QDir::currentPath(),
     QString filename = QFileDialog::getOpenFileName(
       this,
       "Open Document",
       "C:/Users/jun/Documents/livestitching/script/",
       "Document files (*.pto);;All files(*.*)");
    if (!filename.isNull()) { //用户选择了文件
       // 处理文件
       QMessageBox::information(this, "Document", "Has document", QMessageBox::Ok | QMessageBox::Cancel);
       ui->label->setText(filename);
    } else {// 用户取消选择
       QMessageBox::information(this, "Document", "No document", QMessageBox::Ok | QMessageBox::Cancel);
       return;
    }

    QFile readfile(filename);
        if (!readfile.open(QFile::ReadOnly|QIODevice::Text))
            {
                ui->label->setText("open error");
            }
        QTextStream readdata(&readfile);
        QString readline;
        int linenum=0;
        while(!readdata.atEnd()) {
            readline=readdata.readLine();
            if (!readline.isEmpty())
                if (readline.at(0) == 'i'){
                    linenum++;

                    //int heightstart=readline.indexOf(QRegExp("\\bh[\\w=]*\\b"));
                    //int heightend=readline.indexOf(" ",heightstart);
                    //QString heightnum=readline.mid(heightstart,heightend-heightstart);
                    cam[linenum-1].type="fullframe_fisheye";
                    cam[linenum-1].height=getint(readline,"h");
                    cam[linenum-1].width=getint(readline,"w");
                    cam[linenum-1].center_dx=getfloat(readline,"d");
                    cam[linenum-1].center_dy=getfloat(readline,"e");
                    cam[linenum-1].exposure_value=getfloat(readline,"Eev");
                    cam[linenum-1].exposure_value_blue=getfloat(readline,"Eb");
                    cam[linenum-1].exposure_value_red=getfloat(readline,"Er");
                    cam[linenum-1].radial_1=getfloat(readline,"a");
                    cam[linenum-1].radial_2=getfloat(readline,"b");
                    cam[linenum-1].radial_3=getfloat(readline,"c");
                    cam[linenum-1].hfov=QString::number(getfloat(readline,"v").toDouble()/180*3.1415926,'g',15);
                    cam[linenum-1].rotate_1=QString::number(getfloat(readline,"r").toDouble()/180*3.1415926,'g',15);
                    cam[linenum-1].rotate_2=QString::number(-getfloat(readline,"y").toDouble()/180*3.1415926,'g',15);
                    cam[linenum-1].rotate_3=QString::number(-getfloat(readline,"p").toDouble()/180*3.1415926,'g',15);

               }

        }


        readfile.close();

            inputNum=linenum+1;
            ui->spinBox->setMinimum(1);
            ui->spinBox->setMaximum(inputNum-1);
        for (int i=0;i<linenum;i++)
            ui->comboBox_cam->addItem("cam"+QString::number(i));


}

void MainWindow::changeCamInfo(){
    int i=ui->spinBox->value();
    ui->label_width->setNum(cam[i-1].width);
    ui->label_height->setNum(cam[i-1].height);
    ui->label_center_dx->setText(cam[i-1].center_dx);
    ui->label_center_dy->setText(cam[i-1].center_dy);
    ui->label_exp->setText(cam[i-1].exposure_value);
    ui->label_expblue->setText(cam[i-1].exposure_value_blue);
    ui->label_expred->setText(cam[i-1].exposure_value_red);
    ui->label_rotate1->setText(cam[i-1].rotate_1);
    ui->label_rotate2->setText(cam[i-1].rotate_2);
    ui->label_rotate3->setText(cam[i-1].rotate_3);
    ui->label_hfov->setText(cam[i-1].hfov);
    ui->label_type->setText(cam[i-1].type);
    ui->label_radial1->setText(cam[i-1].radial_1);
    ui->label_radial2->setText(cam[i-1].radial_2);
    ui->label_radial3->setText(cam[i-1].radial_3);
}

void MainWindow::overlayinc()
{
    if (overlaynum<10){
        overlaynum++;
        ui->comboBox_cam->addItem("Overlay"+QString::number(overlaynum));
    }
    ui->spinBox_2->setMinimum(1);
    ui->spinBox_2->setMaximum(overlaynum);

    overlay[overlaynum-1].aspect_ratio="0.0";
    overlay[overlaynum-1].cam_opt="0.0";
    overlay[overlaynum-1].rotate_1="0.0";
    overlay[overlaynum-1].rotate_2="0.0";
    overlay[overlaynum-1].rotate_3="0.0";
    ui->spinBox_2->setValue(overlaynum);
}

void MainWindow::overlaydec()
{
    if (overlaynum>0){
        overlaynum--;
        ui->comboBox_cam->removeItem(inputNum+overlaynum-1);
    }
    if (overlaynum==0)
    {
        ui->spinBox_2->setMinimum(0);
        ui->spinBox_2->setMaximum(0);
    }
    else
    {
        ui->spinBox_2->setMaximum(overlaynum);
    }
    overlay[overlaynum].aspect_ratio="0.0";
    overlay[overlaynum].cam_opt="0.0";
    overlay[overlaynum].rotate_1="0.0";
    overlay[overlaynum].rotate_2="0.0";
    overlay[overlaynum].rotate_3="0.0";
    overlay[overlaynum].device=0;
    ui->spinBox_2->setValue(overlaynum);
}

void MainWindow::changeOverlayInfo()
{
    int i=ui->spinBox_2->value();
    if (i==0 || i>overlaynum)
    {
        ui->aspect_ratio->setText("0.0");
        ui->cam_opt->setText("0.0");
        ui->rotate1->setText("0.0");
        ui->rotate2->setText("0.0");
        ui->rotate3->setText("0.0");
    }
    else
    {
        ui->aspect_ratio->setText(overlay[i-1].aspect_ratio);
        ui->cam_opt->setText(overlay[i-1].cam_opt);
        ui->rotate1->setText(overlay[i-1].rotate_1);
        ui->rotate2->setText(overlay[i-1].rotate_2);
        ui->rotate3->setText(overlay[i-1].rotate_3);
    }
}

void MainWindow::changeasp()
{
    int i=ui->spinBox_2->value();
    if (0<i && i<=overlaynum)
    {
        overlay[i-1].aspect_ratio=ui->aspect_ratio->text();
    }
}

void MainWindow::changecamopt()
{
    int i=ui->spinBox_2->value();
    if (0<i && i<=overlaynum)
    {
        overlay[i-1].cam_opt=ui->cam_opt->text();
    }
}

void MainWindow::changerot1()
{
    int i=ui->spinBox_2->value();
    if (0<i && i<=overlaynum)
    {
        overlay[i-1].rotate_1=ui->rotate1->text();
    }
}

void MainWindow::changerot2()
{
    int i=ui->spinBox_2->value();
    if (0<i && i<=overlaynum)
    {
        overlay[i-1].rotate_2=ui->rotate2->text();
    }
}

void MainWindow::changerot3()
{
    int i=ui->spinBox_2->value();
    if (0<i && i<=overlaynum)
    {
        overlay[i-1].rotate_3=ui->rotate3->text();
    }
}


void MainWindow::generatefile()
{
    QString filename = QFileDialog::getSaveFileName(this,
        tr("Save json script"),
        "C:/Users/jun/Documents/livestitching/script/",
        tr("*.json")); //选择路径
    if(filename.isEmpty())
    {
        return;
    }
    QFile file(filename);
    //方式：Append为追加，WriteOnly，ReadOnly
    if (!file.open(QIODevice::WriteOnly|QIODevice::Text)) {
        QMessageBox::critical(NULL, "提示", "无法创建文件");
        return;
    }
    QTextStream out(&file);
    out<<"{"<<endl<<"\t\"inputs\": ["<<endl;
    for (int i=0;i<inputNum-1;i++)
    {
        out<<"\t\t{"<<endl;
        out<<"\t\t\t\"options\": {"<<endl;
        out<<"\t\t\t\t\"height\": "<<cam[i].height<<","<<endl,
        out<<"\t\t\t\t\"center_dy\": "<<cam[i].center_dy<<","<<endl;
        out<<"\t\t\t\t\"width\": "<<cam[i].width<<","<<endl,
        out<<"\t\t\t\t\"center_dx\": "<<cam[i].center_dx<<","<<endl,
        out<<"\t\t\t\t\"exposure_value\": "<<cam[i].exposure_value<<","<<endl,
        out<<"\t\t\t\t\"exposure_value_blue\": "<<cam[i].exposure_value_blue<<","<<endl,
        out<<"\t\t\t\t\"rotate\": ["<<endl;
        out<<"\t\t\t\t\t"<<cam[i].rotate_1<<","<<endl;
        out<<"\t\t\t\t\t"<<cam[i].rotate_2<<","<<endl;
        out<<"\t\t\t\t\t"<<cam[i].rotate_3<<endl;
        out<<"\t\t\t\t],"<<endl;
        out<<"\t\t\t\t\"exposure_value_red\": "<<cam[i].exposure_value_red<<","<<endl,
        out<<"\t\t\t\t\"hfov\": "<<cam[i].hfov<<","<<endl,
        out<<"\t\t\t\t\"radial\": ["<<endl;
        out<<"\t\t\t\t\t"<<cam[i].radial_1<<","<<endl;
        out<<"\t\t\t\t\t"<<cam[i].radial_2<<","<<endl;
        out<<"\t\t\t\t\t"<<cam[i].radial_3<<endl;
        out<<"\t\t\t\t],"<<endl;
        out<<"\t\t\t\t\"mask_points\": []"<<endl;
        out<<"\t\t\t},"<<endl;
        out<<"\t\t\t\"type\": \""<<cam[i].type<<"\""<<endl;
        if (i<inputNum-2)
            out<<"\t\t},"<<endl;
        else
            out<<"\t\t}"<<endl;
    }
    out<<"\t],"<<endl;
    if (overlaynum>0){
        out<<"\t\"overlay\": ["<<endl;
        for (int i=0;i<overlaynum;i++)
        {
            out<<"\t\t{"<<endl;
            out<<"\t\t\t\"type\": \"normal\","<<endl;
            out<<"\t\t\t\"options\": {"<<endl;
            out<<"\t\t\t\t\"aspect_ratio\": "<<overlay[i].aspect_ratio<<","<<endl;
            out<<"\t\t\t\t\"cam_opt\": "<<overlay[i].cam_opt<<","<<endl;
            out<<"\t\t\t\t\"rotate\": ["<<overlay[i].rotate_1<<", "<<overlay[i].rotate_2<<", "<<overlay[i].rotate_3<<"]"<<endl;
            out<<"\t\t\t}"<<endl;
            if (i<overlaynum-1)
                out<<"\t\t},"<<endl;
            else
                out<<"\t\t}"<<endl;
        }
        out<<"\t]"<<endl;
    }
    out<<"\t\"output\": {"<<endl<<"\t\t\"options\": {},"<<endl<<"\t\t\"type\": \"equirectangular\""<<endl<<"\t}"<<endl<<"}"<<endl;
    out.flush();
    file.close();
}



/******************************
********* 打开摄像头 ***********
*******************************/
void MainWindow::openCamera()
{
    QList<QCameraInfo> cameras= QCameraInfo::availableCameras();
    camera->stop();
    delete camera;
    camera=new QCamera(cameras.at(ui->comboBox_device->currentIndex()));
    camera->setViewfinder(viewfinder);
    imageCapture=new QCameraImageCapture(camera);
    //viewfinder->show();
    camera->start();

}


/*************************
********* 拍照 ***********
**************************/
void MainWindow::takingPictures()
{
   imageCapture->capture();
}

/*******************************
***关闭摄像头，释放资源，必须释放***
********************************/
void MainWindow::displayImage(int, QImage image)
{
    ui->label_photo->setPixmap(QPixmap::fromImage(image));
}
void MainWindow::bindCamera()
{
    int i=ui->comboBox_cam->currentIndex();
    int j=ui->comboBox_device->currentIndex();
    //QList<QCameraInfo> cameras= QCameraInfo::availableCameras();

    if (i>=0)
    {
        if (i<inputNum-1)
            cam[i].device=j;
        else
            overlay[i+1-inputNum].device=j;
    }
    else
    {
        qDebug()<<"out of range";
    }

}

void MainWindow::linkcam()
{
    int i=ui->comboBox_cam->currentIndex();
    int j=0;
    if (i>=0)
    {
        if (i<inputNum-1)
            j=cam[i].device;
        else
            j=overlay[i+1-inputNum].device;
    }
    else
    {
        qDebug()<<"out of range";
    }
    if (j!=ui->comboBox_device->currentIndex());
        ui->comboBox_device->setCurrentIndex(j);
}

void MainWindow::generatecommand()
{
    QList<QCameraInfo> cameras= QCameraInfo::availableCameras();
    QString Command="";
    for (int i=0;i<inputNum-1;i++)
        Command=Command+cameras.at(cam[i].device).deviceName()+"  ";
        Command+="\n";
    for (int i=0;i<overlaynum;i++)
        Command=Command+cameras.at(overlay[i].device).deviceName()+"  ";
    ui->textEdit->setText(Command);
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QRegExp double_rx("([0-9]*[\\.][0-9]+)|[0-9]+");

    camera     = NULL;

    /*信号和槽*/

    ui->aspect_ratio->setValidator(new QRegExpValidator(double_rx, ui->aspect_ratio));
    ui->cam_opt->setValidator(new QRegExpValidator(double_rx, ui->cam_opt));
    ui->rotate1->setValidator(new QRegExpValidator(double_rx, ui->rotate1));
    ui->rotate2->setValidator(new QRegExpValidator(double_rx, ui->rotate2));
    ui->rotate3->setValidator(new QRegExpValidator(double_rx, ui->rotate3));
    ui->spinBox_2->setMinimum(0);
    ui->spinBox_2->setMaximum(0);
    ui->spinBox_2->setValue(0);
    inputNum=0;
    overlaynum=0;


    //camera = new QCamera(this);
    viewfinder=new QCameraViewfinder(this);

    ui->horizontalLayout->addWidget(viewfinder);
    ui->label_photo->setScaledContents(true);
    //camera->setViewfinder(viewfinder);

    QList<QCameraInfo> cameras= QCameraInfo::availableCameras();
    camera=new QCamera(cameras.at(0));
    camera->setViewfinder(viewfinder);
    foreach (const QCameraInfo &cameraInfo, cameras){
        qDebug()<<cameraInfo.deviceName()<<endl;
        ui->comboBox_device->addItem(cameraInfo.deviceName());

    }

//    if (cameraInfo.deviceName() == "/dev/video5")
//    {
//        camera->stop();
//        camera->destroyed();
//        camera= new QCamera(cameraInfo);
//        camera->setViewfinder(viewfinder);
        //imageCapture=new QCameraImageCapture(camera);
//    }

    imageCapture=new QCameraImageCapture(camera);
    //viewfinder->show();
    camera->start();

    connect(ui->openfile, &QPushButton::clicked, this, &MainWindow::openPTO);
    connect(ui->generate, &QPushButton::clicked, this, &MainWindow::generatefile);
    connect(ui->spinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &MainWindow::changeCamInfo);
    connect(ui->overlayinc, &QPushButton::clicked, this, &MainWindow::overlayinc);
    connect(ui->overlaydec, &QPushButton::clicked, this, &MainWindow::overlaydec);
    connect(ui->spinBox_2, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &MainWindow::changeOverlayInfo);
    connect(ui->aspect_ratio, &QLineEdit::editingFinished,this, &MainWindow::changeasp);
    connect(ui->cam_opt, &QLineEdit::editingFinished,this, &MainWindow::changecamopt);
    connect(ui->rotate1, &QLineEdit::editingFinished,this, &MainWindow::changerot1);
    connect(ui->rotate2, &QLineEdit::editingFinished,this, &MainWindow::changerot2);
    connect(ui->rotate3, &QLineEdit::editingFinished,this, &MainWindow::changerot3);
    connect(ui->comboBox_device,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),this,&MainWindow::openCamera);
    //connect(imageCapture,&QCameraImageCapture::imageCaptured,this,&MainWindow::displayImage);
//    connect(ui->opencam,&QPushButton::clicked, this, &MainWindow::openCamera);
    //connect(ui->capturecam, &QPushButton::clicked, this, &MainWindow::takingPictures);
    connect(ui->bindcam, &QPushButton::clicked, this, &MainWindow::bindCamera);
    connect(ui->comboBox_cam,static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),this,&MainWindow::linkcam);
    connect(ui->run,&QPushButton::clicked, this,&MainWindow::generatecommand);



}

MainWindow::~MainWindow()
{
    delete ui;
}
