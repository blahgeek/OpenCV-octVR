/* 
* @Author: BlahGeek
* @Date:   2016-02-22
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-22
*/

#include <iostream>

#include <QHEaderView>
#include <QFileDialog>

#include "./pto_template.hpp"

PTOTemplate::PTOTemplate(QTreeView * _tree): tree_view(_tree) {
    this->json_model.setEditableFields(QStringList({
        "yaw", "roll", "pitch", "aspect_ratio", "cam_opt",
    }));
    tree_view->setModel(&json_model);
    tree_view->header()->setSectionResizeMode(QHeaderView::ResizeToContents);

    connect(&json_model, &QJsonModel::dataChanged, this, &PTOTemplate::dataChanged);
}

void PTOTemplate::loadPTO() {
    QString filename = QFileDialog::getOpenFileName(nullptr, "Open Document",
                                                    "/home",
                                                    "Document files (*.pto);;All files(*.*)");
    if(filename.isNull())
        return;

    QFile parser_script_file(":/scripts/ptx2json.py");
    parser_script_file.open(QFile::ReadOnly);
    QString parser_script = parser_script_file.readAll();
    parser_script_file.close();

    QProcess parser;
    parser.start("python3",
                 QStringList({"-c", parser_script, filename}));
    parser.waitForFinished();
    QString parsed_json = parser.readAllStandardOutput();
    this->json_model.loadJson(parsed_json.toUtf8());
}

QJsonDocument PTOTemplate::getJsonDocument() {
    return json_model.document();
}
