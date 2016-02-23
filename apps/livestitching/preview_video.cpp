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

PreviewVideoWidget::PreviewVideoWidget(QWidget * parent, int _w, int _h): 
QVideoWidget(parent), preview_w(_w), preview_h(_h) {
    preview_data0.setKey(OCTVR_PREVIEW_DATA0_MEMORY_KEY);
    preview_data1.setKey(OCTVR_PREVIEW_DATA1_MEMORY_KEY);
    preview_meta.setKey(OCTVR_PREVIEW_DATA_META_MEMORY_KEY);

#define CREATE(x, s) \
    do { \
        qDebug() << "Creating shared memory with size = " << (s); \
        bool ret = (x).create(s); \
        if(!ret) \
            qDebug() << (x).errorString(); \
        CV_Assert(ret); \
    } while(0)

    CREATE(preview_data0, sizeof(struct PreviewDataHeader) + _w * _h * 3);
    CREATE(preview_data1, sizeof(struct PreviewDataHeader) + _w * _h * 3);
    CREATE(preview_meta, 1);

    *(static_cast<char *>(preview_meta.data())) = 0;
}

void PreviewVideoWidget::paintEvent(QPaintEvent *) {
    char * p_index = static_cast<char *>(preview_meta.data());
    *p_index = 1 - *p_index;

    QPainter painter(this);
    QSharedMemory & data = (*p_index == 1 ? preview_data0 : preview_data1);
    data.lock();
    struct PreviewDataHeader * hdr = static_cast<struct PreviewDataHeader *>(data.data());
    CV_Assert(hdr->width == preview_w);
    CV_Assert(hdr->height == preview_h);
    QImage img(static_cast<unsigned char *>(data.data()) + sizeof(struct PreviewDataHeader), 
               preview_w, preview_h, preview_w * 3, QImage::Format_RGB32);
    painter.drawImage(QPoint(0, 0), img);
    data.unlock();
}

void PreviewVideoWidget::updatePreview() {
    this->repaint();
}
