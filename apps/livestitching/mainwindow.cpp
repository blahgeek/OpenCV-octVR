#if defined(_WIN32)
#define Q_COMPILER_INITIALIZER_LISTS value
#endif
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
#include <QProcess>
#include <QDebug>

void MainWindow::openPTO() {
    QString filename = QFileDialog::getOpenFileName(this, "Open Document", 
                                                    "C:/Users/jun/Documents/livestitching/script/", 
                                                    "Document files (*.pto);;All files(*.*)");
    if(filename.isNull())
        return;
    ui->label_filename->setText(filename);

    QFile parser_script_file(":/scripts/ptx2json.py");
    parser_script_file.open(QFile::ReadOnly);
    QString parser_script = parser_script_file.readAll();
    parser_script_file.close();

    QProcess parser;
    parser.start("python3",
                 QStringList({"-c", parser_script, filename}));
    parser.waitForFinished();
    QString parsed_json = parser.readAllStandardOutput();
    this->json_model.loadJson(parsed_json.toUtf8());

    for(int i = 0 ; i < this->overlay_cameras.size() ; i += 1)
        this->jsonAddOverlay();
}

void MainWindow::jsonAddOverlay() {
    QJsonDocument doc = json_model.document();

    QFile sample_file(":/data/sample_overlay.json");
    sample_file.open(QFile::ReadOnly);
    QString sample = sample_file.readAll();
    sample_file.close();

    QJsonDocument sample_doc = QJsonDocument::fromJson(sample.toUtf8());
    QJsonObject doc_obj = doc.object();
    QJsonArray overlays = doc_obj["overlays"].toArray();
    overlays.append(sample_doc.object());
    doc_obj.insert("overlays", overlays);

    doc.setObject(doc_obj);
    json_model.loadDocument(doc);
}

void MainWindow::jsonDelOverlay() {
    QJsonDocument doc = json_model.document();

    QJsonObject doc_obj = doc.object();
    QJsonArray overlays = doc_obj["overlays"].toArray();
    overlays.pop_back();
    doc_obj.insert("overlays", overlays);

    doc.setObject(doc_obj);
    json_model.loadDocument(doc);
}

void MainWindow::deviceAddCamera() {
    bool is_overlay = ui->comboBox_input_overlay->currentIndex() > 0;
    const QCameraInfo & current_cam_info = camera_infos.at(ui->comboBox_device->currentIndex());

    QCamera * camera = new QCamera(current_cam_info);
    QCameraViewfinder * viewfinder = new QCameraViewfinder(this);
    camera->setViewfinder(viewfinder);
    camera->start();

    if(is_overlay) {
        ui->layout_overlays->addWidget(viewfinder);
        overlay_cameras.emplace_back(std::unique_ptr<QCamera>(camera), 
                                     std::unique_ptr<QCameraViewfinder>(viewfinder),
                                     current_cam_info.deviceName());
        this->jsonAddOverlay();
    }
    else {
        ui->layout_inputs->addWidget(viewfinder);
        input_cameras.emplace_back(std::unique_ptr<QCamera>(camera), 
                                   std::unique_ptr<QCameraViewfinder>(viewfinder),
                                   current_cam_info.deviceName());
    }
}

void MainWindow::deviceDelCamera() {
    bool is_overlay = ui->comboBox_input_overlay->currentIndex() > 0;
    std::vector<MainWindow::CameraAndView> & cameras = is_overlay ? overlay_cameras : input_cameras;
    if(cameras.empty())
        return;

    std::get<0>(cameras.back())->stop();
    cameras.pop_back();

    if(is_overlay)
        this->jsonDelOverlay();
}

void MainWindow::run() {
    if(json_model.document().object()["inputs"].toArray().size() != input_cameras.size()) {
        QMessageBox::warning(this, "Bad Template", "Input count does not match");
        return;
    }
    if(json_model.document().object()["overlays"].toArray().size() != overlay_cameras.size()) {
        QMessageBox::warning(this, "Bad Template", "Overlay count does not match");
        return;
    }
    if(input_cameras.size() + overlay_cameras.size() == 0) {
        QMessageBox::warning(this, "Bad Template", "No input");
        return;
    }

    QString temp_path = QDir::tempPath();

    QString output_json_path = temp_path + "/vr.json";
    QFile output_json(output_json_path);
    output_json.open(QIODevice::WriteOnly);
    output_json.write(json_model.document().toJson());
    output_json.close();

    QStringList dumper_args;
    dumper_args << "-w" << QString::number(ui->paranoma_width->value())
                << "-o" << (temp_path + "/vr.dat")
                << output_json_path;
    qDebug() << dumper_args.join(" ");

    QStringList args;
    for(auto & input: input_cameras)
        args << "-f" << "v4l2" << "-input_format" << "rgb24"
             << "-framerate" << "30" << "-i" << std::get<2>(input);
    for(auto & overlay: overlay_cameras)
        args << "-f" << "v4l2" << "-input_format" << "rgb24"
             << "-framerate" << "30" << "-i" << std::get<2>(overlay);
    args << "-filter_complex" 
         << QString("vr_map=inputs=%1:outputs=:crop_x=%2:crop_w=%3:blend=%4")
            .arg(input_cameras.size() + overlay_cameras.size())
            .arg(ui->paranoma_crop_x->value())
            .arg(ui->paranoma_crop_w->value())
            .arg(ui->paranoma_algorithm->currentIndex() == 0 ? 128 : -5);
    args << "-c:v" << ui->encoding_codec->currentText()
         << "-b:v" << QString("%1M").arg(ui->encoding_bitrate->value())
         << "-g" << QString::number(ui->encoding_gopsize->value());
    args << "-hls_time" << QString::number(ui->hls_segment_time->value())
         << "-hls_list_size" << QString::number(ui->hls_list_size->value())
         << "-hls_flags" << "delete_segments"
         << "-hls_allow_cache" << "0"
         << "-y" << QString("%1/vr.m3u8").arg(ui->hls_dir->text());
    qDebug() << args.join(" ");
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->json_model.setEditableFields(QStringList({
        "yaw", "roll", "pitch", "aspect_ratio", "cam_opt", 
    }));
    ui->treeView->setModel(&json_model);
    ui->treeView->header()->setSectionResizeMode(QHeaderView::ResizeToContents);

    connect(ui->pushButton_openfile, &QPushButton::clicked, this, &MainWindow::openPTO);

    // Cameras
    camera_infos= QCameraInfo::availableCameras();
    foreach(const QCameraInfo &info, camera_infos) {
        qDebug() << "Camera Info: " << info.description() << ", " << info.deviceName();
        ui->comboBox_device->addItem(info.description());
    }
    connect(ui->pushButton_add_camera, &QPushButton::clicked, this, &MainWindow::deviceAddCamera);
    connect(ui->pushButton_del_camera, &QPushButton::clicked, this, &MainWindow::deviceDelCamera);

    connect(ui->pushButton_run, &QPushButton::clicked, this, &MainWindow::run);

}

MainWindow::~MainWindow()
{
    delete ui;
}
