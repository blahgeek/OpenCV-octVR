/* 
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-23
*/

#include <iostream>

#include "./runner.hpp"
#include <QDebug>
#include <QMessageBox>

#include <assert.h>

Runner::Runner() {
    connect(&dumper_proc, static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, &Runner::onDumperProcessFinished);
    connect(&ffmpeg_proc, static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this, &Runner::onFfmpegProcessFinished);
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

void Runner::start(QStringList _dumper_args, QStringList _ffmpeg_args) {
    this->dumper_args = _dumper_args;
    this->ffmpeg_args = _ffmpeg_args;

    if(this->status() != Runner::NOT_RUNNING) {
        qDebug() << "Runner already running, return";
        return;
    }

    qDebug() << "Running dumper: " << dumper_args;
    dumper_proc.start("/home/blahgeek/dumper", dumper_args); // FIXME
    emit statusChanged();
    // bool finished = dumper_proc.waitForFinished();
    // if(!(finished && dumper_proc.exitStatus() == QProcess::NormalExit && dumper_proc.exitCode() == 0)) {
    //     QMessageBox::warning(nullptr, "", "Unable to create dat file");
    //     this->onRunningStatusChanged(NOT_RUNNING);
    //     return;
    // }
}

void Runner::stop() {
    ffmpeg_proc.terminate();
}

void Runner::onDumperProcessFinished(int exitCode, QProcess::ExitStatus status) {
    if(status != QProcess::NormalExit || exitCode != 0) {
        qDebug() << "Dumper did not finish normally";
        QMessageBox::warning(nullptr, "", "Unable to create dat file");
        emit statusChanged();
        return;
    }
    // run ffmpeg

    qDebug() << "Running ffmpeg: " << ffmpeg_args;
    ffmpeg_proc.start("/home/blahgeek/ffmpeg", ffmpeg_args); // FIXME

    emit statusChanged();
}

void Runner::onFfmpegProcessFinished(int exitCode, QProcess::ExitStatus status) {
    QMessageBox::warning(nullptr, "", QString("Stopped: %1 %2").arg(status).arg(exitCode));
    emit statusChanged();
}
