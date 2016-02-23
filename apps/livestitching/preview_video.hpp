/* 
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-23
*/

#ifndef LIVESTITCHING_PREVIEW_VIDEO_H__
#define LIVESTITCHING_PREVIEW_VIDEO_H__ value

#include <QVideoWidget>
#include <QSharedMemory>

#include "octvr.hpp"

class PreviewVideoWidget: public QVideoWidget {

    Q_OBJECT

private:
    QSharedMemory preview_data0, preview_data1, preview_meta;
    int preview_w, preview_h;

protected:
    void paintEvent(QPaintEvent *event);

public:
    PreviewVideoWidget(QWidget * parent=nullptr,
                       int preview_w=1280, int preview_h=640);

public slots:
    void updatePreview();

};

#endif
