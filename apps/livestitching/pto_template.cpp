/* 
* @Author: BlahGeek
* @Date:   2016-02-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-11
*/

#include <iostream>
#include <QHeaderView>
#include <QFileDialog>
#include <QCoreApplication>

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
        double arc_per_cam = 360.0 / lon_select_num;
        if(this->left)
            parser_args << QString("--lon_select=%1,%2,%3,%4").arg(-arc_per_cam * 0.2).arg(arc_per_cam * 1.2).arg(-arc_per_cam).arg(lon_select_num);
        else
            parser_args << QString("--lon_select=%1,%2,%3,%4").arg(-arc_per_cam * 1.2).arg(arc_per_cam * 0.2).arg(-arc_per_cam).arg(lon_select_num);
    }
    parser_args << filename;

    qDebug() << "Parser args: " << parser_args;

    QProcess parser;
#if defined ( _WIN32 )
    parser.start("\"" + QCoreApplication::applicationDirPath() + "/python3/python3\"", parser_args);
#else
    parser.start("python3", parser_args);
#endif

    parser.waitForFinished();
    QString parsed_json = parser.readAllStandardOutput();
    this->json_model.loadJson(parsed_json.toUtf8());
}

QJsonDocument PTOTemplate::getJsonDocument() {
    return json_model.document();
}
