/* 
* @Author: BlahGeek
* @Date:   2016-02-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-22
*/

#ifndef LIVESTITCHING_TEMPLATE_H__
#define LIVESTITCHING_TEMPLATE_H__ value


#include "./qjsonmodel.h"

#include <QTreeView>

class PTOTemplate: public QObject {

    Q_OBJECT

private:
    QTreeView * tree_view;
    QJsonModel json_model;

public:
    PTOTemplate(QTreeView * _treeview);

public slots:
    void loadPTO();

signals:
    void dataChanged();

public:
    QJsonDocument getJsonDocument();
};

#endif
