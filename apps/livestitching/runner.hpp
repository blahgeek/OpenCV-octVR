/* 
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-27
*/

#ifndef LIVESTITCHING_RUNNER_H__
#define LIVESTITCHING_RUNNER_H__ value

#include <QProcess>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <QJsonDocument>

#include <utility>
#include "octvr.hpp"

class Runner : public QObject {
    Q_OBJECT

private:
    std::vector<std::pair<QJsonDocument, cv::Size>> json_queues;
    QProcess dumper_proc, ffmpeg_proc;
    QString ffmpeg_args;

    QTemporaryDir temp_dir;

public:
    enum RunningStatus { NOT_RUNNING, DUMPER_RUNNING, FFMPEG_RUNNING };

    enum RunningStatus status() const;
    void start(std::vector<std::pair<QJsonDocument, cv::Size>> json_docs,
               QString _ffmpeg_args);

    Runner();

public slots:
    void onDumperProcessFinished(int exitCode, QProcess::ExitStatus status);
    void onFfmpegProcessFinished(int exitCode, QProcess::ExitStatus status);
    void stop();

signals:
    void statusChanged();

};

#endif
