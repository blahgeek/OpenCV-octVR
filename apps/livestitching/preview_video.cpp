/* 
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-23
*/

#include <iostream>

#include "./preview_video.hpp"
#include "octvr.hpp"

#include <QPainter>
#include <QDebug>

using vr::PreviewDataHeader;

static void createSharedMemory(QSharedMemory & x, QString name, size_t s) {
    // FIXME, UGLY
    qDebug() << "Creating memory " << name;
    x.setKey(name);
    qDebug() << "Native name: " << x.nativeKey();
    x.create(s);
    if(x.error() == QSharedMemory::NoError) {
        qDebug() << "Created, done";
        return;
    }
    else if(x.error() == QSharedMemory::AlreadyExists) {
        qDebug() << "Already exists, attach and detach...";
        x.attach();
        x.detach();
        x.create(s);
        CV_Assert(x.error() == QSharedMemory::NoError);
        qDebug() << "done.";
    } else {
        qDebug() << "Error: " << x.errorString();
        CV_Assert(false);
    }
}

PreviewVideoWidget::PreviewVideoWidget(QWidget * parent, int _w, int _h): 
QVideoWidget(parent), preview_w(_w), preview_h(_h) {
    createSharedMemory(preview_data0, OCTVR_PREVIEW_DATA0_MEMORY_KEY,
                       sizeof(struct PreviewDataHeader) + _w * _h * 3);
    createSharedMemory(preview_data1, OCTVR_PREVIEW_DATA1_MEMORY_KEY,
                       sizeof(struct PreviewDataHeader) + _w * _h * 3);
    createSharedMemory(preview_meta, OCTVR_PREVIEW_DATA_META_MEMORY_KEY, 1);

    *(static_cast<char *>(preview_meta.data())) = 0;
    memset(preview_data0.data(), 0, sizeof(struct PreviewDataHeader));
    memset(preview_data1.data(), 0, sizeof(struct PreviewDataHeader));
}

void PreviewVideoWidget::paintEvent(QPaintEvent *) {
    std::lock_guard<std::mutex> lock(this->mtx);

    char * p_index = static_cast<char *>(preview_meta.data());
    *p_index = 1 - *p_index;

    qDebug() << "Previewing data at " << int(1 - *p_index);

    QPainter painter(this);
    QSharedMemory & data = (*p_index == 1 ? preview_data0 : preview_data1);
    data.lock();
    struct PreviewDataHeader * hdr = static_cast<struct PreviewDataHeader *>(data.data());
    if(hdr->width != 0) {
        CV_Assert(hdr->width == preview_w);
        CV_Assert(hdr->height == preview_h);
        qDebug() << "Drawing...";
        QImage img(static_cast<unsigned char *>(data.data()) + sizeof(struct PreviewDataHeader), 
                   preview_w, preview_h, preview_w * 3, QImage::Format_RGB888);
        painter.drawImage(QRect(QPoint(0, 0), this->sizeHint()), img);
    }
    data.unlock();
}

void PreviewVideoWidget::updatePreview() {
    this->update();
}
