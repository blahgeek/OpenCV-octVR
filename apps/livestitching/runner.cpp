/*
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-05-12
*/

#include <iostream>

#include "./runner.hpp"
#include "./encryptor.hpp"
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
    if(dumper_running || !json_queues.empty())
        return DUMPER_RUNNING;
    if(ffmpeg_running)
        return FFMPEG_RUNNING;
    return NOT_RUNNING;
}

void Runner::start(std::vector<std::pair<QJsonDocument, cv::Size>> json_docs,
                   QString _ffmpeg_args) {
    if(this->status() != Runner::NOT_RUNNING) {
        qDebug() << "Runner already running, return";
        return;
    }

    this->json_queues = json_docs;
    this->ffmpeg_args = Encryptor::encryptArgString(_ffmpeg_args);

    emit statusChanged();
    this->onDumperProcessFinished(0, QProcess::NormalExit); // to start dumper
}

void Runner::stop() {
    ffmpeg_proc.kill();
}

void Runner::onDumperProcessFinished(int exitCode, QProcess::ExitStatus status) {
    if(status != QProcess::NormalExit || exitCode != 0) {
        QMessageBox::warning(nullptr, "", "Unable to create dat file, bad template?");
        json_queues.clear();
        emit statusChanged();
        return;
    }

    if(json_queues.empty()) { // dump ok
        QString _run = "\"" + QCoreApplication::applicationDirPath() + "/OwlLiveCore\""
                          + " " + ffmpeg_args;
        qDebug() << "Running OwlLiveCore: " << _run;
        ffmpeg_proc.start(_run);
        emit statusChanged();
        return;
    }

    auto this_json = json_queues.back();
    json_queues.pop_back();

    QString json_path = temp_dir.path() + QDir::separator() + QString::number(json_queues.size()) + ".json";
    QString dat_path = temp_dir.path() + QDir::separator() + QString::number(json_queues.size()) + ".dat";
    QFile json_f(json_path);
    json_f.open(QIODevice::WriteOnly);
    json_f.write(this_json.first.toJson());
    json_f.close();
    QString dumper = QCoreApplication::applicationDirPath() + "/octvr_dump";
    dumper_proc.start(dumper, QStringList({"-w", QString::number(this_json.second.width), 
                                           "-h", QString::number(this_json.second.height),
                                           "-o", dat_path, json_path}));

    emit statusChanged();
}

void Runner::onFfmpegProcessFinished(int exitCode, QProcess::ExitStatus status) {
    QMessageBox::warning(nullptr, "", QString("Stitcher stopped unexpectedly (%1 %2)").arg(status).arg(exitCode));
    emit statusChanged();
}
