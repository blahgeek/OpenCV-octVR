/* 
* @Author: BlahGeek
* @Date:   2016-02-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-30
*/

#include <iostream>
#include <cmath>

#include <QFileDialog>
#include <QProcess>
#include <QMessageBox>
#include <QCameraViewfinderSettings>
#include <QCoreApplication>

#include "./inputs_select.hpp"

InputsSelector::InputsSelector(QGridLayout * _grid): grid(_grid) {
    this->camera_infos = QCameraInfo::availableCameras();

    int size_all = this->camera_infos.size();
    int size_row = std::ceil(std::sqrt(float(size_all)));

#if defined(__linux__)
   QCameraViewfinderSettings settings;
   settings.setMaximumFrameRate(5);
   settings.setResolution(QSize(640, 480));
#endif

    int row = 0, col = 0;
    foreach(const QCameraInfo &info, camera_infos) {
        qDebug() << "Camera info: " << info.description() << ", " << info.deviceName();

        this->cameras.emplace_back(new QCamera(info));
        this->views.emplace_back(new QCameraViewfinder());
        this->cameras.back()->setViewfinder(this->views.back().get());
        // this->cameras.back()->setViewfinderSettings(settings);
        this->cameras.back()->start();

        QGroupBox * group_box = new QGroupBox(info.description());
        QVBoxLayout * layout = new QVBoxLayout;
        layout->addWidget(this->views.back().get());

        this->check_boxs.emplace_back(new QCheckBox("Select"));
        this->check_boxs.back()->setCheckState(Qt::Checked);
        connect(this->check_boxs.back().get(), &QCheckBox::stateChanged,
                this, &InputsSelector::selectedChanged);
        layout->addWidget(this->check_boxs.back().get());

        group_box->setLayout(layout);

        this->grid->addWidget(group_box, row, col);
        col += 1;
        if(col == size_row) {
            row += 1;
            col = 0;
        }
    }
}

void InputsSelector::start() {
    for(auto & c: cameras)
        c->start();
}

void InputsSelector::stop() {
    for(auto & c: cameras)
        c->stop();
}

void InputsSelector::onInputsFpsChanged(int _fps) {
    qDebug() << "Inputs FPS is set to " << _fps;
    this->fps = _fps;
}

std::vector<QCameraInfo> InputsSelector::getSelected() {
    std::vector<QCameraInfo> ret;
    for(int i = 0 ; i < this->camera_infos.size() ; i += 1)
        if(this->check_boxs[i]->checkState() == Qt::Checked)
            ret.push_back(this->camera_infos.at(i));
    return ret;
}

std::vector<QCameraInfo> InputsSelector::getAll() {
    std::vector<QCameraInfo> ret;
    for(int i = 0 ; i < this->camera_infos.size() ; i += 1)
        ret.push_back(this->camera_infos.at(i));
    return ret;
}

QStringList InputsSelector::getInputArgs() {
    QStringList args;

    auto all_cams = this->getAll();
    auto selected_cams = this->getSelected();

#if defined(_WIN32)
    for(auto & input: selected_cams) {
        int device_name_dup = 0;
        for(auto & x: all_cams) {
            if(x.deviceName() == input.deviceName()) // it's me
                break;
            if(x.description() == input.description()) // dup
                device_name_dup += 1;
        }
        args << "-f" << "dshow" << "-pixel_format" << "uyvy422"
             << "-video_device_number" << QString::number(device_name_dup)
             << "-framerate" << QString::number(this->fps)
             << "-i" << QString("video=%1").arg(input.description());
    }
#else
    for(auto & input: selected_cams)
        args << "-f" << "v4l2" << "-pixel_format" << "uyvy422"
             << "-framerate" << QString::number(this->fps) << "-i" << input.deviceName();
#endif

    return args;
}

void InputsSelector::saveImages(int crop_x, int crop_w) {
    QString dir = QFileDialog::getExistingDirectory(nullptr, ("Choose Directory"),
                                                    "/home", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    QStringList in_args = this->getInputArgs();
    QStringList out_args;

    auto selected = this->getSelected();
    for(size_t i = 0 ; i < selected.size() ; i += 1) {
        out_args << "-map" << QString("%1").arg(i);
        if(crop_w > 0)
            out_args << "-vf" << QString("crop=w=%1:x=%2").arg(crop_w).arg(crop_x);
        out_args << "-vframes" << "1"
                 << "-y" << QString("%1/crop_\%d_%2.bmp").arg(dir).arg(i);
    }
    qDebug() << "Running: " << in_args << out_args;

    QProcess proc;
    proc.start(QCoreApplication::applicationDirPath() + "/ffmpeg",
               in_args + out_args);
    bool finished = proc.waitForFinished();
    if(finished && proc.exitStatus() == QProcess::NormalExit && proc.exitCode() == 0)
        QMessageBox::information(nullptr, "", "Images saved");
    else
        QMessageBox::warning(nullptr, "", "Error occured while saving images");
}
