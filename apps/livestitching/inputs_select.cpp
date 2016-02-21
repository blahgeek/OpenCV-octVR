/* 
* @Author: BlahGeek
* @Date:   2016-02-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-22
*/

#include <iostream>
#include <cmath>

#include <QFileDialog>

#include "./inputs_select.hpp"

InputsSelector::InputsSelector(QGridLayout * _grid): grid(_grid) {
    this->camera_infos = QCameraInfo::availableCameras();

    int size_all = this->camera_infos.size();
    int size_row = std::ceil(std::sqrt(float(size_all)));

    int row = 0, col = 0;
    foreach(const QCameraInfo &info, camera_infos) {
        qDebug() << "Camera info: " << info.description() << ", " << info.deviceName();

        this->cameras.emplace_back(new QCamera(info));
        this->views.emplace_back(new QCameraViewfinder());
        this->cameras.back()->setViewfinder(this->views.back().get());
        this->cameras.back()->start();

        QGroupBox * group_box = new QGroupBox(info.description());
        QVBoxLayout * layout = new QVBoxLayout;
        layout->addWidget(this->views.back().get());

        this->check_boxs.emplace_back(new QCheckBox("Enable"));
        this->check_boxs.back()->setCheckState(Qt::Checked);
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

std::vector<QCameraInfo> InputsSelector::getSelected() {
    std::vector<QCameraInfo> ret;
    for(int i = 0 ; i < this->camera_infos.size() ; i += 1)
        if(this->check_boxs[i]->checkState() == Qt::Checked)
            ret.push_back(this->camera_infos.at(i));
    return ret;
}

void InputsSelector::saveImages() {
    QString dir = QFileDialog::getExistingDirectory(nullptr, ("Choose Directory"),
                                                    "/home", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    auto selected = this->getSelected();
    QStringList in_args, out_args;
    for(size_t i = 0 ; i < selected.size() ; i += 1) {
        in_args << "-f" << "v4l2" << "-input_format" << "rgb24"
                << "-i" << selected[i].deviceName();
        out_args << "-map" << QString("%1").arg(i)
                 // TODO
                 // << "-vf" << QString("crop=w=%1:x=%2")
                 //         .arg(ui->paranoma_crop_w->value())
                 //         .arg(ui->paranoma_crop_x->value())
                 << "-vframes" << "1"
                 << "-y" << QString("crop_\%d_%1.bmp").arg(i);
    }
    qDebug() << "Running: " << in_args << out_args;
    // TODO
}
