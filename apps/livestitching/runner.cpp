/*
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   StrayWarrior
* @Last Modified time: 2016-03-18
*/

#include <iostream>

#include "./runner.hpp"
#include <QDebug>
#include <QMessageBox>
#include <QCoreApplication>

#include <assert.h>
#include <sodium.h>

static QString encryptArgString(QString _str) {
#ifndef OWLLIVE_ENCRYPT_ARG
    return _str;
#else
    // Modify the encrypt method here.
    const unsigned char secret[] = {103, 246, 81, 250, 242, 200, 201, 94, 240,
                                    238, 74, 26, 34, 3, 148, 59, 107, 95, 189,
                                    173, 111, 120, 101, 65, 74, 154, 28, 96,
                                    200, 247, 247, 52};
    if (sodium_init() == -1) {
        return QString();
    }
    QByteArray _str_bytes;
    _str_bytes.append(_str);
    const unsigned char * _str_cptr = reinterpret_cast<const unsigned char *>(_str_bytes.data());
    unsigned int _str_len = _str.length();

    unsigned char nonce[crypto_secretbox_NONCEBYTES];
    randombytes_buf(nonce, sizeof nonce);

    unsigned int cipher_len = crypto_secretbox_MACBYTES + _str_len;
    unsigned char * cipher_str = new unsigned char[cipher_len];

    if (crypto_secretbox_easy(cipher_str, _str_cptr, _str_len, nonce, secret) != 0){
        return QString();
    }

    QByteArray ret_bytes = QByteArray(reinterpret_cast<const char *>(nonce), crypto_secretbox_NONCEBYTES);
    ret_bytes.append(reinterpret_cast<const char *>(cipher_str), cipher_len);

    QString ret = ret_bytes.toBase64();
    delete cipher_str;

    return ret;
#endif
}

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
    // if necessary, encrypt the arguments
    ffmpeg_args = encryptArgString(ffmpeg_args);

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
