/* 
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-05
*/

#include <iostream>

#include "./preview_video.hpp"
#include "octvr.hpp"

#include <QPainter>
#include <QDebug>

using vr::PreviewDataHeader;

static bool createSharedMemory(QSharedMemory & x, QString name, size_t s, bool retry=true) {
    if(x.isAttached()) {
        qDebug() << "Detaching shared memory " << name << " before creating";
        x.detach();
    }
    qDebug() << "Creating shared memory with name=" << name << " and size=" << s;
    x.setKey(name);
    bool ret = x.create(s);
    if(ret) {
        qDebug() << "Success created shared memory";
        return true;
    }
    if (!retry || x.error() != QSharedMemory::AlreadyExists) {
        qDebug() << "Cannot create shared memory: " << x.errorString();
        return false;
    }
    qDebug() << "Shared memory already exists, attach and detatch and retry";
    x.attach();
    x.detach();
    return createSharedMemory(x, name, s, false);
}

PreviewVideoWidget::PreviewVideoWidget(QWidget * p): QVideoWidget(p) {}

void PreviewVideoWidget::prepare(int _w, int _h) {
    std::lock_guard<std::mutex> lock(this->mtx);
    valid_shared_memory = true;
    if(_w * _h == 0) {
        qDebug() << "Preview video is disabled";
        valid_shared_memory = false;
    } else {
        valid_shared_memory &= createSharedMemory(preview_data0, OCTVR_PREVIEW_DATA0_MEMORY_KEY,
                                                  sizeof(struct PreviewDataHeader) + _w * _h * 3);
        valid_shared_memory &= createSharedMemory(preview_data1, OCTVR_PREVIEW_DATA1_MEMORY_KEY,
                                                  sizeof(struct PreviewDataHeader) + _w * _h * 3);
        valid_shared_memory &= createSharedMemory(preview_meta, OCTVR_PREVIEW_DATA_META_MEMORY_KEY, 1);
    }

    if(valid_shared_memory) {
        *(static_cast<char *>(preview_meta.data())) = 0;
        memset(preview_data0.data(), 0, sizeof(struct PreviewDataHeader));
        memset(preview_data1.data(), 0, sizeof(struct PreviewDataHeader));
    }
    else {
        qDebug() << "Shared memory not ready!";
    }
}

void PreviewVideoWidget::paintEvent(QPaintEvent *) {
    std::lock_guard<std::mutex> lock(this->mtx);
    QPainter painter(this);

    if(!valid_shared_memory) {
        painter.drawText(QPointF(5, 10), "Preview not available");
        return;
    }

    char * p_index = static_cast<char *>(preview_meta.data());
    *p_index = 1 - *p_index;

    qDebug() << "Previewing data at " << int(1 - *p_index);

    QSharedMemory & data = (*p_index == 1 ? preview_data0 : preview_data1);
    data.lock();
    struct PreviewDataHeader * hdr = static_cast<struct PreviewDataHeader *>(data.data());
    if(hdr->width != 0) {
        CV_Assert(hdr->width == preview_w);
        CV_Assert(hdr->height == preview_h);
        qDebug() << "Drawing...";
        QImage img(static_cast<unsigned char *>(data.data()) + sizeof(struct PreviewDataHeader), 
                   preview_w, preview_h, preview_w * 3, QImage::Format_RGB888);
        painter.drawImage(QRect(QPoint(0, 0), this->size()), img);
        painter.drawText(QPointF(5, 10), QString("FPS: %1").arg(QString::number(hdr->fps, 'g', 4)));
    }
    data.unlock();
}

void PreviewVideoWidget::updatePreview() {
    this->update();
}
