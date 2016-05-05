/*
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-18
*/

#include <iostream>

#include "./runner.hpp"
#include "./encryptor.hpp"
#include <QDebug>
#include <QMessageBox>
#include <QCoreApplication>

#include <assert.h>

Runner::Runner() {
    connect(&dumper_proc_left, static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, &Runner::onDumperProcessFinished);
    connect(&dumper_proc_right, static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, &Runner::onDumperProcessFinished);
    connect(&ffmpeg_proc, static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, &Runner::onFfmpegProcessFinished);

    assert(temp_dir.isValid());
    qDebug() << "Temporary dir: " << temp_dir.path();

    dumper_proc_left.setWorkingDirectory(temp_dir.path());
    dumper_proc_right.setWorkingDirectory(temp_dir.path());
    ffmpeg_proc.setWorkingDirectory(temp_dir.path());
}

enum Runner::RunningStatus Runner::status() const {
    bool dumper_running = (dumper_proc_left.state() != QProcess::NotRunning || 
                           dumper_proc_right.state() != QProcess::NotRunning);
    bool ffmpeg_running = ffmpeg_proc.state() != QProcess::NotRunning;
    assert(!(dumper_running && ffmpeg_running));
    if(dumper_running)
        return DUMPER_RUNNING;
    if(ffmpeg_running)
        return FFMPEG_RUNNING;
    return NOT_RUNNING;
}

void Runner::start(QJsonDocument json_doc_left, 
                   QJsonDocument json_doc_right,
                   int width,
                   QString _ffmpeg_args) {
    this->ffmpeg_args = _ffmpeg_args;

    if(this->status() != Runner::NOT_RUNNING) {
        qDebug() << "Runner already running, return";
        return;
    }

    auto dump_json = [&, this](const QJsonDocument & doc, const char * filename) {
        QString output_json_path = temp_dir.path() + QDir::separator() + filename;
        QFile output_json(output_json_path);
        output_json.open(QIODevice::WriteOnly);
        output_json.write(doc.toJson());
        output_json.close();
    };

    QString dumper = QCoreApplication::applicationDirPath() + "/octvr_dump";

    dump_json(json_doc_left, "left.json");
    dumper_proc_left.start(dumper, QStringList({"-w", QString::number(width), "-o", "left.dat", "left.json"}));

    if(!json_doc_right.isNull()) {
        dump_json(json_doc_right, "right.json");
        dumper_proc_right.start(dumper, QStringList({"-w", QString::number(width), "-o", "right.dat", "right.json"}));
    }

    emit statusChanged();
}

void Runner::stop() {
    ffmpeg_proc.kill();
}

void Runner::onDumperProcessFinished(int exitCode, QProcess::ExitStatus status) {
    if(status != QProcess::NormalExit || exitCode != 0) {
        qDebug() << "Dumper did not finish normally";
        QMessageBox::warning(nullptr, "", "Unable to create dat file, bad template?");
        emit statusChanged();
        return;
    }
    if(this->status() != NOT_RUNNING)
        return;
    // run ffmpeg
    // if necessary, encrypt the arguments
    ffmpeg_args = Encryptor::encryptArgString(ffmpeg_args);

    QString _run = "\"" + QCoreApplication::applicationDirPath() + "/OwlLiveCore\""
                      + " " + ffmpeg_args;
    qDebug() << "Running ffmpeg: " << _run;
    ffmpeg_proc.start(_run);

    emit statusChanged();
}

void Runner::onFfmpegProcessFinished(int exitCode, QProcess::ExitStatus status) {
    QMessageBox::warning(nullptr, "", QString("Stitcher stopped unexpectedly (%1 %2)").arg(status).arg(exitCode));
    emit statusChanged();
}
