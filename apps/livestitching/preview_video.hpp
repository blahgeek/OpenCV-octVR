/* 
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-05
*/

#ifndef LIVESTITCHING_PREVIEW_VIDEO_H__
#define LIVESTITCHING_PREVIEW_VIDEO_H__ value

#include <QVideoWidget>
#include <QSharedMemory>

#include "octvr.hpp"

#include <mutex>

class PreviewVideoWidget: public QVideoWidget {

    Q_OBJECT

private:
    QSharedMemory preview_data0, preview_data1, preview_meta;
    int preview_w = 0, preview_h = 0;

    bool valid_shared_memory = false;

    std::mutex mtx;

protected:
    void paintEvent(QPaintEvent *event);

public:
    PreviewVideoWidget(QWidget * parent=nullptr);
    bool isValid() { return valid_shared_memory; }
    void prepare(int _w, int _h);

public slots:
    void updatePreview();

};

#endif
