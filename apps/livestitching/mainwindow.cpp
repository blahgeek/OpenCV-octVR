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

void MainWindow::loadPTO(const QString & filename) {
    if(filename.isNull())
        return;

    const QString pto_dest_filename = temp_path + "/template.pto";
    // Copy template file to temporary dir.
    if (QFile::exists(pto_dest_filename)) {
        if (QFile::remove(pto_dest_filename)) {
            QMessageBox::critical(this, "Error", "No permission to " + pto_dest_filename);
            return;
        }
    }
    QFile::copy(filename, pto_dest_filename);

    QFile parser_script_file(":/scripts/ptx2json.py");
    parser_script_file.open(QFile::ReadOnly);
    QString parser_script = parser_script_file.readAll();
    parser_script_file.close();

    QProcess parser;
    parser.start("python3",
                 QStringList({"-c", parser_script, pto_dest_filename}));
    parser.waitForFinished();
    QString parsed_json = parser.readAllStandardOutput();
    this->json_model.loadJson(parsed_json.toUtf8());

    for(size_t i = 0 ; i < this->overlay_cameras.size() ; i += 1)
        this->jsonAddOverlay();
    return;
}

void MainWindow::openPTO() {
    QString filename = QFileDialog::getOpenFileName(this, "Open Document",
                                                    "C:/Users/jun/Documents/livestitching/script/",
                                                    "Document files (*.pto);;All files(*.*)");
    loadPTO(filename);
}

void MainWindow::locateHugin() {
    QString filename = QFileDialog::getOpenFileName(this, "Locate Hugin",
                                                    "C:/Users/jun/Documents/livestitching/script/",
                                                    "Hugin executable file (*.*);;All files(*.*)");
    hugin_path = filename;
    // FIXME
    // ui->path_hugin->setText(filename);
}

void MainWindow::reEditPTO() {
    QFileInfo hugin = QFileInfo(hugin_path);
    if (!hugin.exists()) {
        QMessageBox::warning(this, "File not exist", "Cannot find hugin");
    }
    QProcess hugin_proc;
    hugin_proc.start(hugin.absoluteFilePath(), QStringList{ temp_path + "/template.pto" });
    hugin_proc.waitForFinished();

    return;
}

void MainWindow::saveAsPTO() {
    QString filename = QFileDialog::getSaveFileName(this, "Save Template",
                                                    "C:/Users/jun/Documents/livestitching/script/",
                                                    "Document files (*.pto);;All files(*.*)");
    if (QFile::exists(filename)) {
        if (QFile::remove(filename)) {
            QMessageBox::critical(this, "Error", "No permission to" + filename);
            return;
        }
    }
    QFile::copy(temp_path + "/template.pto", filename);
    return;
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
        viewfinder->setFixedSize(200, 200);
        viewfinder->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
        this->inputs_layout->addWidget(viewfinder);
        // ui->layout_inputs->addWidget(viewfinder);
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

void MainWindow::gotoStitch() {
    // Make sure the dir exists.
    QDir stitch_dir(temp_path);
    if (!stitch_dir.mkpath(".")){
        QMessageBox::critical(this, "Error", "Cannot create temporary dir");
        return;
    }

    QImageEncoderSettings image_setting;
    image_setting.setCodec("image/jpeg");
    image_setting.setQuality(QMultimedia::VeryHighQuality);

    QStringList image_list;
    QStringList crop_in_args, crop_out_args;

    // Capture pictures from all input cameras. (Asynchronous)
    // TODO: Should it replaced by FFmpeg-VR ?
    for (size_t i = 0; i < input_cameras.size(); ++i){
        auto && camera = std::get<0>(input_cameras[i]);
        QString camera_info = std::get<2>(input_cameras[i]);

        QString image_file_name = QString("crop_\%d_%1.bmp").arg(i);
        crop_in_args << "-f" << "v4l2" << "-input_format" << "rgb24"
                     << "-i" << camera_info;
        crop_out_args << "-map" << QString("%1").arg(i)
                      << "-vf" << QString("crop=w=%1:x=%2")
                         .arg(ui->inputs_crop_w->value())
                         .arg(ui->inputs_crop_x->value())
                      << "-vframes" << "1"
                      << image_file_name;
        image_list << image_file_name;
        /* using FFmpeg-VR instead.
         * Do not delete these codes. They may be used in Windows!
        QCameraImageCapture * image_capture = new QCameraImageCapture(camera.get());
        image_capture->setEncodingSettings(image_setting);
        if (image_capture->isReadyForCapture()){
            QString image_file_name = QString("%1/%2.jpg").arg(temp_path).arg(i);
            int capture_id = image_capture->capture(image_file_name);
            image_captures.insert(std::make_pair(capture_id, image_capture));
            image_list << image_file_name;
            connect(image_capture, &QCameraImageCapture::imageSaved, this, &MainWindow::removeImageCapture);
        }
        */
    }

    QProcess ffmpeg_crop_proc;
    // QFileInfo ffmpeg = QFileInfo(ui->path_ffmpeg->text());
    QFileInfo ffmpeg = QFileInfo("./ffmpeg");  // FIXME
    if (!ffmpeg.exists()) {
        QMessageBox::warning(this, "File not exist", "Cannot find ffmpeg");
        return;
    }
    ffmpeg_crop_proc.setWorkingDirectory(temp_path);
    QStringList crop_args;
    crop_args << crop_in_args << crop_out_args << "-y";
    qDebug() << crop_args;
    ffmpeg_crop_proc.start(ffmpeg.absoluteFilePath(), crop_args);
    ffmpeg_crop_proc.waitForFinished();
    QString ffmpeg_error = ffmpeg_crop_proc.readAllStandardError();
    qDebug() << ffmpeg_error;

    // Create template.pto file
    // QFileInfo pto_gen = QFileInfo(ui->path_pto_gen->text());
    QFileInfo pto_gen = QFileInfo("./pto_gen"); // FIXME
    if (!pto_gen.exists()) {
        QMessageBox::warning(this, "File not exist", "Cannot find pto_gen");
        return;
    }
    QProcess pto_gen_proc;
    QStringList pto_gen_args;
    pto_gen_proc.setWorkingDirectory(temp_path);
    // TODO: The parameters should not be fixed.
    pto_gen_args << "-o" << "template.pto" << "-p" << "3" << "-f" << "120" << image_list;
    pto_gen_proc.start(pto_gen.absoluteFilePath(), pto_gen_args);
    pto_gen_proc.waitForFinished();

    // Run hugin.
    reEditPTO();

    loadPTO(temp_path.append("/template.pto"));

    return;
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

    // QFileInfo dumper = QFileInfo(ui->path_dumper->text());
    QFileInfo dumper = QFileInfo("./dumper"); // FIXME
    if (!dumper.exists()){
        QMessageBox::warning(this, "File not exist", "Cannot find dumper");
        return;
    }
    QProcess dumper_proc;
    dumper_proc.start(dumper.absoluteFilePath(), dumper_args);
    dumper_proc.waitForFinished();
    qDebug() << "Template dumped.";

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
            .arg(ui->inputs_crop_x->value())
            .arg(ui->inputs_crop_w->value())
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

    // QFileInfo ffmpeg = QFileInfo(ui->path_ffmpeg->text());
    QFileInfo ffmpeg = QFileInfo("./ffmpeg"); // FIXME
    if (!ffmpeg.exists()){
        QMessageBox::warning(this, "File not exist", "Cannot find ffmpeg");
        return;
    }

    this->ffmpeg_proc.start(ffmpeg.absoluteFilePath(), args);

    return;
}

void MainWindow::initPreview() {
    videoWidget = new QVideoWidget;
    ui->preview_vlayout->addWidget(videoWidget);

    videoPreviewer = new QMediaPlayer;
    videoPreviewer->setVideoOutput(videoWidget);
    videoWidget->show();

    connect(ui->pushButton_preview, &QPushButton::clicked, this, &MainWindow::startPreview);

    return;
}

void MainWindow::startPreview() {
    videoPreviewer->setMedia(QUrl::fromUserInput(ui->preview_url->text()));
    videoPreviewer->play();
    ui->pushButton_preview->setText("Stop");

    connect(ui->pushButton_preview, &QPushButton::clicked, this, &MainWindow::stopPreview);
    return;
}

void MainWindow::stopPreview() {
    videoPreviewer->stop();
    ui->pushButton_preview->setText("Play");

    connect(ui->pushButton_preview, &QPushButton::clicked, this, &MainWindow::startPreview);
    return;
}

void MainWindow::onTabChanged(int index) {
    qDebug() << "onTabChanged: " << index;
    if(index == 0)
        this->inputs_selector->start();
    else {
        this->inputs_selector->stop();
        auto selected = this->inputs_selector->getSelected();
        qDebug() << "selected count: " << selected.size();
    }
}

void MainWindow::onInputsSelectChanged() {
    auto selected = this->inputs_selector->getSelected();
    auto all = this->inputs_selector->getAll();
    this->ui->inputs_info->setText(QString("%1 found, %2 selected")
                                   .arg(all.size())
                                   .arg(selected.size()));
    this->ui->inputs_action_save->setEnabled(selected.size() > 0);
}

void MainWindow::onInputSaveButtonClicked() {
    this->inputs_selector->saveImages(this->ui->inputs_crop_x->value(),
                                      this->ui->inputs_crop_w->value());
}

void MainWindow::onTemplateChanged() {
    auto json_doc = this->pto_template->getJsonDocument();
    QJsonObject doc_obj = json_doc.object();
    QJsonArray inputs = doc_obj["inputs"].toArray();
    this->ui->template_info->setText(QString("%1 inputs").arg(inputs.size()));
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    inputs_selector.reset(new InputsSelector(ui->inputs_grid));
    pto_template.reset(new PTOTemplate(ui->template_tree_view));

    inputs_layout = new QHBoxLayout(ui->area_inputs_widget);
    ui->area_inputs_widget->setFixedSize(0, 0);
    ui->area_inputs_widget->setSizePolicy(QSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum));
    ui->area_inputs->setWidgetResizable(true);
    // inputs_layout->setSpacing(1);

    {
        QFile f(":qdarkstyle/style.qss");
        Q_ASSERT(f.exists());
        f.open(QFile::ReadOnly | QFile::Text);
        QTextStream ts(&f);
        qApp->setStyleSheet(ts.readAll());
    }

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

    connect(ui->pushButton_new_pto, &QPushButton::clicked, this, &MainWindow::gotoStitch);
    connect(ui->pushButton_edit_pto, &QPushButton::clicked, this, &MainWindow::reEditPTO);
    connect(ui->pushButton_saveas_pto, &QPushButton::clicked, this, &MainWindow::saveAsPTO);
    connect(ui->pushButton_locate_hugin, &QPushButton::clicked, this, &MainWindow::locateHugin);

    connect(ui->pushButton_run, &QPushButton::clicked, this, &MainWindow::run);

    connect(ui->tabWidget, &QTabWidget::currentChanged, this, &MainWindow::onTabChanged);
    connect(ui->inputs_action_save, &QPushButton::clicked, this, &MainWindow::onInputSaveButtonClicked);
    connect(this->inputs_selector.get(), &InputsSelector::selectedChanged, this, &MainWindow::onInputsSelectChanged);

    connect(ui->template_load, &QPushButton::clicked, this->pto_template.get(), &PTOTemplate::loadPTO);
    connect(this->pto_template.get(), &PTOTemplate::dataChanged, this, &MainWindow::onTemplateChanged);

    this->onInputsSelectChanged();
    this->onTemplateChanged();

    temp_path = QDir::tempPath() + "/vrlive";
    qDebug() << "temporary dir: " << temp_path;

}

MainWindow::~MainWindow()
{
    delete ui;
}
