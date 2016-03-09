/* 
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-06
*/

#include <iostream>

#include "./runner.hpp"
#include <QDebug>
#include <QMessageBox>
#include <QCoreApplication>

#include <assert.h>

Runner::Runner() {
    connect(&dumper_proc, static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, &Runner::onDumperProcessFinished);
    connect(&ffmpeg_proc, static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, &Runner::onFfmpegProcessFinished);

    assert(temp_dir.isValid());
    qDebug() << "Temporary dir: " << temp_dir.path();

    dumper_proc.setWorkingDirectory(temp_dir.path());
    ffmpeg_proc.setWorkingDirectory(temp_dir.path());
}

enum Runner::RunningStatus Runner::status() const {
    bool dumper_running = dumper_proc.state() != QProcess::NotRunning;
    bool ffmpeg_running = ffmpeg_proc.state() != QProcess::NotRunning;
    assert(!(dumper_running && ffmpeg_running));
    if(dumper_running)
        return DUMPER_RUNNING;
    if(ffmpeg_running)
        return FFMPEG_RUNNING;
    return NOT_RUNNING;
}

void Runner::start(QJsonDocument json_doc, int width,
                   QString _ffmpeg_args) {
    this->ffmpeg_args = _ffmpeg_args;

    if(this->status() != Runner::NOT_RUNNING) {
        qDebug() << "Runner already running, return";
        return;
    }

    QString output_json_path = temp_dir.path() + "/vr.json";
    QFile output_json(output_json_path);
    output_json.open(QIODevice::WriteOnly);
    output_json.write(json_doc.toJson());
    output_json.close();

    QStringList dumper_args;
    dumper_args << "-w" << QString::number(width)
                << "-o" << "vr.dat"
                << output_json_path;

    qDebug() << "Running dumper: " << dumper_args;
    dumper_proc.start(QCoreApplication::applicationDirPath() + "/octvr_dump", 
                      dumper_args);
    emit statusChanged();
}

void Runner::stop() {
    ffmpeg_proc.kill();
}

void Runner::onDumperProcessFinished(int exitCode, QProcess::ExitStatus status) {
    if(status != QProcess::NormalExit || exitCode != 0) {
        qDebug() << "Dumper did not finish normally";
        QMessageBox::warning(nullptr, "", "Unable to create dat file");
        emit statusChanged();
        return;
    }
    // run ffmpeg

    QString _run = "\"" + QCoreApplication::applicationDirPath() + "/ffmpeg\""
                      + " " + ffmpeg_args;
    qDebug() << "Running ffmpeg: " << _run;
    ffmpeg_proc.start(_run);

    emit statusChanged();
}

void Runner::onFfmpegProcessFinished(int exitCode, QProcess::ExitStatus status) {
    QMessageBox::warning(nullptr, "", QString("Stopped: %1 %2").arg(status).arg(exitCode));
    emit statusChanged();
}
