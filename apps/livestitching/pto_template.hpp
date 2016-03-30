/* 
* @Author: BlahGeek
* @Date:   2016-02-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-30
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
    int lon_select_num = 0;
    bool left;

public:
    PTOTemplate(QTreeView * _treeview, bool _left);

public slots:
    void loadPTO();

signals:
    void dataChanged();

public:
    QJsonDocument getJsonDocument();
    void setLonSelectionNumber(int _n){ lon_select_num = _n; }
};

#endif
