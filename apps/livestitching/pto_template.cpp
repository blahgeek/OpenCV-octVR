/* 
* @Author: BlahGeek
* @Date:   2016-02-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-30
*/

#include <iostream>
#if defined(_WIN32)
#define Q_COMPILER_INITIALIZER_LISTS value
#endif
#include <QHeaderView>
#include <QFileDialog>

#include "./pto_template.hpp"

PTOTemplate::PTOTemplate(QTreeView * _tree, bool _left): tree_view(_tree), left(_left) {
    this->json_model.setEditableFields(QStringList({
        "yaw", "roll", "pitch", "aspect_ratio", "cam_opt",
    }));
    tree_view->setModel(&json_model);
    tree_view->header()->setSectionResizeMode(QHeaderView::ResizeToContents);

    connect(&json_model, &QJsonModel::dataChanged, this, &PTOTemplate::dataChanged);
}

void PTOTemplate::loadPTO() {
    static QString filename = QDir::homePath();
    filename = QFileDialog::getOpenFileName(nullptr, "Open Template",
                                            filename,
                                            "Hugin template (*.pto);;PTGui template (*.pts);;All files(*.*)");
    if(filename.isNull()){
        filename = QDir::homePath();
        return;
    }

    QFile parser_script_file(":/scripts/ptx2json.py");
    parser_script_file.open(QFile::ReadOnly);
    QString parser_script = parser_script_file.readAll();
    parser_script_file.close();

    QStringList parser_args;
    parser_args << "-c" << parser_script;
    if(this->lon_select_num > 0) {
        parser_args << "--lon_select";
        if(this->left)
            parser_args << QString("%1,%2,%3,%4").arg(-3.0).arg(360.0 / lon_select_num + 3.0).arg(360.0 / lon_select_num).arg(lon_select_num);
        else
            parser_args << QString("%1,%2,%3,%4").arg(- 360.0 / lon_select_num - 3.0).arg(3.0).arg(360.0 / lon_select_num).arg(lon_select_num);
    }
    parser_args << filename;

    QProcess parser;
    parser.start("python3", parser_args);
    parser.waitForFinished();
    QString parsed_json = parser.readAllStandardOutput();
    this->json_model.loadJson(parsed_json.toUtf8());
}

QJsonDocument PTOTemplate::getJsonDocument() {
    return json_model.document();
}
