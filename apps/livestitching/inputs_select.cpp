/* 
* @Author: BlahGeek
* @Date:   2016-02-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-05-12
*/

#include <iostream>
#include <cmath>

#include <QFileDialog>
#include <QProcess>
#include <QMessageBox>
#include <QCameraViewfinderSettings>
#include <QCoreApplication>
#include <QRegularExpression>
#include <QAudio>

#include "./inputs_select.hpp"
#include "./encryptor.hpp"

#include <algorithm>

static long get_camera_order(const QCameraInfo & info) {
    long ret = 0;

    QRegularExpression re("([0-9]+)");
    QRegularExpressionMatchIterator it = re.globalMatch(info.description());
    while(it.hasNext()) {
        ret <<= 8;
        QRegularExpressionMatch m = it.next();
        ret += m.captured(1).toInt();
    }
    return ret;
}

InputsSelector::InputsSelector(QGridLayout * _grid, QComboBox * _audio_combo): 
grid(_grid), audio_combo(_audio_combo) {

    // QUICK HACK for integrated camera
    auto all_cam_infos = QCameraInfo::availableCameras();
    foreach(const QCameraInfo & info, all_cam_infos)
        if(!info.description().contains("BisonCam", Qt::CaseInsensitive))
            this->camera_infos.append(info);

    std::stable_sort(camera_infos.begin(), camera_infos.end(), [](const QCameraInfo & a, const QCameraInfo & b) {
        return get_camera_order(a) < get_camera_order(b);
    });

    int size_all = this->camera_infos.size();
    int size_row = std::ceil(std::sqrt(float(size_all)));

    int row = 0, col = 0;
    foreach(const QCameraInfo &info, camera_infos) {
        qDebug() << "Camera info: " << info.description() << ", " << info.deviceName() << ", " << get_camera_order(info);

        this->cameras.emplace_back(new QCamera(info));
        this->views.emplace_back(new QCameraViewfinder());
        this->cameras.back()->setViewfinder(this->views.back().get());

#if defined(__linux__)
        this->cameras.back()->load();
        auto supported_settings = this->cameras.back()->supportedViewfinderSettings();
        QCameraViewfinderSettings settings;
        foreach(const QCameraViewfinderSettings &s,supported_settings) {
            if(s.resolution().width() <= 800 && (settings.isNull() || settings.maximumFrameRate() > s.maximumFrameRate()))
                settings = s;
        }
        if(!settings.isNull()) {
            qDebug() << "Setting camera to " << settings.resolution() << "@" <<
                settings.minimumFrameRate() << ":" << settings.maximumFrameRate();
            this->cameras.back()->setViewfinderSettings(settings);
        }
#endif
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

    this->audio_infos = QAudioDeviceInfo::availableDevices(QAudio::AudioInput);
    foreach(const QAudioDeviceInfo &info, audio_infos) {
        qDebug() << "Audio info: " << info.deviceName();
        audio_combo->addItem(info.deviceName());
    }
}

void InputsSelector::start() {
    for(auto & c: cameras)
        c->start();
}

void InputsSelector::stop() {
    for(auto & c: cameras) {
        c->stop();
        c->unload();
    }
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

QStringList InputsSelector::getInputArgs(int width, int height) {
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
             << "-video_size" << QString("%1x%2").arg(width).arg(height)
             << "-video_device_number" << QString::number(device_name_dup)
             << "-framerate" << QString::number(this->fps)
             << "-i" << QString("video=%1").arg(input.description());
    }
#else
    for(auto & input: selected_cams)
        args << "-f" << "v4l2" << "-pixel_format" << "uyvy422"
             << "-video_size" << QString("%1x%2").arg(width).arg(height)
             << "-framerate" << QString::number(this->fps) << "-i" << input.deviceName();
#endif

    if(this->audio_combo->currentIndex() != 0) {
    #if defined(_WIN32)
        args << "-f" << "dshow" << "-i" << QString("audio=%1").arg(this->audio_combo->currentText());
    #else
        args << "-f" << "alsa" << "-i" << this->audio_combo->currentText();
    #endif
        args << "-strict" << "-2";
    }

    return args;
}

void InputsSelector::saveImages(int width, int height, int crop_x, int crop_w) {
    for(auto & c: cameras) {
        c->stop();
        c->unload();
    }

    QString dir = QFileDialog::getExistingDirectory(nullptr, ("Choose Directory"),
                                                    "/home", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    QStringList in_args = this->getInputArgs(width, height);
    QStringList out_args;

    auto selected = this->getSelected();
    for(size_t i = 0 ; i < selected.size() ; i += 1) {
        out_args << "-map" << QString("%1").arg(i);
        if(crop_w > 0)
            out_args << "-vf" << QString("crop=w=%1:x=%2").arg(crop_w).arg(crop_x);
        out_args << "-vsync" << "drop" << "-vframes" << "1"
                 << "-y" << QString("%1/crop_\%d_%2.bmp").arg(dir).arg(i);
    }
    qDebug() << "Running: " << in_args << out_args;

    QProcess proc;
    QString encrypted_args = Encryptor::encryptArgString(Encryptor::concatArgString(in_args + out_args));
    qDebug() << encrypted_args;
    proc.start("\"" + QCoreApplication::applicationDirPath() + "/OwlLiveCore\" " + encrypted_args);
    bool finished = proc.waitForFinished();
    if(finished && proc.exitStatus() == QProcess::NormalExit && proc.exitCode() == 0)
        QMessageBox::information(nullptr, "", "Images saved");
    else
        QMessageBox::warning(nullptr, "", "Error occured while saving images");

    for(auto & c: cameras)
        c->start();
}
